export VWModel, VWClassifier, VWRegressor

@with_kw mutable struct VWModel <: BaseEstimator
    vw::Vector{UInt8} = UInt8[]
    l1::Float32 = 0
    l2::Float32 = 0
    learning_rate::Float32 = 0.5
    nn::Int = 0
    interactions = ""
    passes::Int = 10
    loss_function::String = "squared"
    link::String = "identity"
    cmd::String = ""
end

is_classifier(m::VWModel) = m.loss_function == "logistic"

support_multiclass(::VWModel) = false

function gridparams(m::VWModel)
    grid = [
        "l2" => [0.001, 0.01, 0.1, 1, 10],
        "learning_rate" => [0.01, 0.1, 0.5],
        [
            ["nn" => [0, 10, 100]], 
            ["interactions" => ["", "ff"]]
        ]
    ]
end

function fit!(m::VWModel, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack vw, l1, l2, learning_rate, passes = m
    @unpack nn, interactions, loss_function, link, cmd = m
    dst = tovw(x, y, w, shuffle = true)
    args = split(cmd, ' ')
    nn > 0 && push!(args, "--nn", "$nn")
    !isempty(interactions) && push!(args, "--interactions", interactions)
    run(`$VW $dst -c -f vw --l1 $l1 --l2 $l2 --passes $passes
         --loss_function $loss_function -l $learning_rate $args`)
    run(`$VW $dst -t --quiet -i vw --audit_regressor audit.txt`)
    # run(pipeline(`$VW_VARINFO $dst -t -i $path`, stdout = "varinfo.txt"))
    m.vw = read("vw")
    foreach(rm, ["vw", dst, dst * ".cache"])
    BSON.bson("model.bson", model = m)
    visualize(m, columns)
    return m
end

function predict_vw(m::VWModel, x)
    @unpack vw, link = m
    write("vw", vw)
    dst = tovw(x, shuffle = false)
    run(`$VW $dst -t -i vw -p preds --link $link`)
    ŷ = readdlm("preds", Float32)
    foreach(rm, [dst, "vw", "preds"])
    return ŷ
end

predict_proba(m::VWModel, x) = predict_vw(m, x)

function predict(m::VWModel, x)
    @unpack loss_function = m
    ŷ = predict_vw(m, x)
    if loss_function == "logistic"
        ŷ .= sign.(ŷ .- 0.5)       
    end
    return ŷ
end

function tovw(x, y = nothing, w = nothing; shuffle = false)
    x = reshape(x, size(x, 1), :)
    y = isnothing(y) ? ones(size(x, 2)) : y
    w = isnothing(w) ? ones(size(x, 2)) : w
    if memory(x) > 1024
        h5 = @sprintf("/dev/shm/%s.h5", randstring())
        dst = @sprintf("/dev/shm/%s.vw", randstring())
        h5save(h5, (x = x, y = y, w = w))
        hdf2vw = joinpath(@__DIR__, "hdf2vw.jl")
        run(`$JULIA --startup-file=no -p 10 $hdf2vw
            --shuffle $shuffle --dst $dst $h5`)
        rm(h5, force = true)
    else
        x = reshape(x, size(x, 1), :)
        dst = @sprintf("/dev/shm/%s.vw", randstring())
        fid = open(dst, "w")
        js = shuffle ? randperm(length(y)) : 1:length(y)
        @showprogress "tovw..." for j in js
            print(fid, y[j], ' ')
            print(fid, w[j], " |f ")
            @inbounds for i in 1:size(x, 1)
                @printf(fid, "%d:%.4g ", i - 1, x[i, j])
            end
            println(fid)
        end
        close(fid)
    end
    return dst
end

function visualize(m::VWModel, columns)
    mapper = Dict(string(n - 1) => c for (n, c) in enumerate(columns))
    mapper["Constant"] = "Constant"
    if isfile("varinfo.txt")
        df = pd.read_csv("varinfo.txt", delimiter = raw"\s+")
        df["FeatureName"] = map(df["FeatureName"]) do x
            join([mapper[split(c, '^')[end]] for c in split(x, '*')], '*')
        end
        df.to_csv("varinfo.csv", index = false, encoding = "gbk")
    end
    if isfile("audit.txt")
        df = pd.read_csv("audit.txt", delimiter = ":", header = nothing)
        df.columns = ["FeatureName", "HashVal", "Weight"]
        df["FeatureName"] = map(df["FeatureName"]) do x
            join([mapper[split(c, '^')[end]] for c in split(x, '*')], '*')
        end
        df["RelScore"] = 100 * df["Weight"] / df["Weight"].abs().max()
        df.sort_values("Weight", inplace = true)
        df.to_csv("audit.csv", index = false, encoding = "gbk")
    end
end

modelhash(m::VWModel) = hash(m.vw)

VWClassifier(;ka...) = VWModel(;loss_function = "logistic", link = "logistic", ka...)

VWRegressor(;ka...) = VWModel(;loss_function = "squared", link = "identity", ka...)