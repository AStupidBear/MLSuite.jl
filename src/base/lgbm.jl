export LgbmModel, LgbmRegressor, LgbmClassifier

@with_kw mutable struct LgbmModel <: BaseEstimator
    pyo::PyObject = PyNULL()
    so::Vector{UInt8} = UInt8[]
    max_bin::Int = 16
    objective::String = "binary:logistic"
    warm_start::Bool = false
    num_leaves::Int = 31
    min_child_samples::Int = 1000
    n_estimators::Int = 50
    reg_lambda::Float32 = 0.1
    subsample::Float32 = 1
    colsample_bytree::Float32 = 1
end

is_classifier(m::LgbmModel) = occursin(r"binary|multi", m.objective)

function paramgrid(m::LgbmModel)
    grid = OrderedDict(
        "num_leaves" => [15, 31],
        "min_child_samples" => [1000, 100, 5000],
        "n_estimators" => [20, 50],
        "reg_lambda" => [0.1, 1, 10],
        "subsample" => [1, 0.8],
        "colsample_bytree" => [1, 0.8],
    )
    params = paramgrid(grid)
end

function fit!(m::LgbmModel, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack max_bin, objective, warm_start, num_leaves, min_child_samples = m
    @unpack n_estimators, reg_lambda, subsample, colsample_bytree = m
    dump_lgbm_data(x, y, w, max_bin = max_bin, columns = columns)
    if occursin(r"binary|multi", objective)
        num_class = length(unique(y))
    end
    if occursin("binary", objective) && num_class > 2
        objective = "multi:softmax"
    end
    min_child_samples = min(length(y) ÷ 10, min_child_samples)
    # get node lists
    nodes = unique(pmap(n -> gethostname(), workers()))
    nnode, hosts = length(nodes), join(nodes, ',')
    # make cmd args
    args = String[]
    push!(args, "task=train")
    objective = denormalize_objective(objective, "lightgbm")
    push!(args, "objective=$objective")
    occursin("multiclass", objective) &&
    push!(args, "num_class=$num_class")
    push!(args, "tree_learner=data")
    push!(args, "n_estimators=$n_estimators")
    push!(args, "n_jobs=20")
    push!(args, "train_data=train.lgbm")
    push!(args, "num_machines=$nnode")
    push!(args, "output_model=lgbm")
    warm_start && isfile("lgbm") &&
    push!(args, "input_model=lgbm")
    push!(args, "num_leaves=$num_leaves")
    push!(args, "min_child_samples=$min_child_samples")
    push!(args, "reg_lambda=$reg_lambda")
    push!(args, "subsample=$subsample")
    push!(args, "colsample_bytree=$colsample_bytree")
    if isnothing(Sys.which("mpiexec"))
        run(`$LIGHTGBM $args`)
    else
        run(`mpiexec --host $hosts $LIGHTGBM $args`)
    end
    @from lightgbm imports Booster
    @from treelite imports Model
    pyo = Booster(model_file = "lgbm")
    em = eachmatch(r"Tree=(\d+)", read("lgbm", String))
    ntree = collect(em)[end].captures[1]
    if parse(Int, ntree) > 0
        model = Model.load("lgbm", model_format = "lightgbm")
        rm("lgbm.so", force = true)
        model.export_lib(toolchain = "gcc", libpath = "lgbm.so")
        so = read("lgbm.so")
    else
        so = UInt8[]
    end
    @pack! m = pyo, so
    visualize(m, columns)
    return m
end

function predict_lgbm(m::LgbmModel, x; treelite)
    @unpack pyo, so = m
    if treelite && !isempty(so) 
        write("lgbm.so", so)
        @imports treelite.runtime as rt
        batch = rt.Batch.from_npy2d(pymat(x))
        pyo = rt.Predictor("lgbm.so")
        pyo.predict(batch)
    else
        pyo.predict(pymat(x))
    end
end

function predict_proba(m::LgbmModel, x; treelite = false)
    ŷ = predict_lgbm(m, x, treelite = treelite)
    ndims(ŷ) == 1 ? hcat(1 .- ŷ,  ŷ) : ŷ
end

function predict(m::LgbmModel, x; treelite = false)
    @unpack objective = m
    ŷ = predict_lgbm(m, x, treelite = treelite)
    !is_classifier(m) && return ŷ
    if ndims(ŷ) == 1
        signone.(ŷ .- 0.5)
    else
        [I[2] - 1 for I in argmax(ŷ, dims = 2)]
    end
end

function visualize(m::LgbmModel, columns)
    @unpack pyo = m
    gbm = GbmModel(pyo = pyo)
    visualize(gbm, columns)
end

reset!(m::LgbmModel) = foreach(c -> rm(c, force = true), ["train.lgbm", "lgbm", "lgbm.so", "ref.bson"])

istree(m::LgbmModel) = true

modelhash(m::LgbmModel) = hash(m.pyo)

function dump_lgbm_data(x, y, w = nothing; max_bin = 16, columns)
    w = isnothing(w) ? fill!(similar(y), 1) : w
    if isfile("ref.bson") && isfile("train.lgbm")
        BSON.@load "ref.bson" xseg yoff woff
    else
        xseg, yoff, woff = zeros(Float32), -1, -1
    end
    if xseg == x[1:min(end, 100000)] && length(unique(y)) <= 2
        open("train.lgbm", "r+") do fid
            @assert yoff > 0
            seek(fid, yoff - 1)
            write(fid, y)
            if woff > 0
                seek(fid, woff - 1)
                write(fid, w)
            end
        end
    else
        xseg = x[1:min(end, 100000)]
        x, y, w = pymat(x), vec(y), vec(w)
        @from lightgbm imports Dataset
        @from unidecode imports unidecode
        dset = Dataset(data = x, label = y, weight = w, params = Dict("max_bin" => max_bin - 1))
        columns′ = [filter(!isspace, unidecode(c)) for c in columns]
        dset.set_feature_name(columns′)
        isfile("train.lgbm") && rm("train.lgbm")
        @time dset.save_binary("train.lgbm")
        @gc dset
        data = Mmap.mmap("train.lgbm")
        yoff = align(data, reinterpret(UInt8, y))
        woff = align(data, reinterpret(UInt8, w))
        BSON.@save "ref.bson" xseg yoff woff
    end
end

LgbmRegressor(;ka...) = LgbmModel(;objective = "reg:mse", ka...)

LgbmClassifier(;ka...) = LgbmModel(;objective = "binary:logistic", ka...)