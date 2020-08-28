export H2oModel, H2oRegressor, H2oClassifier

@with_kw mutable struct H2oModel <: BaseEstimator
    pyo::PyObject = PyNULL()
    isclf::Bool = true
    name::String = "glm"
    family::String = "gaussian"
    lambda_search::Bool = true
    alpha::Float32 = 0
    gamma::Float32 = -1
    hyper_param::Float32 = 1
    ntrees::Int = 50
    sample_rate::Float32 = 1
    col_sample_rate::Float32 = 1
    col_sample_rate_per_tree::Float32 = 1
    max_depth::Int = 5
    min_rows::Int = 1
    score_tree_interval::Int = 5
    stopping_rounds::Int = 5
    learn_rate::Float32 = 0.1
    epochs::Int = 10
    adaptive_rate::Bool = true
    activation::String = "rectifier_with_dropout"
    rho::Float32 = 0.99
    epsilon::Float32 = 1e-8
    input_dropout_ratio::Float32 = 0
    hidden::Vector = [100, 100]
    hidden_dropout_ratios::Vector = [0.5, 0.5]
    max_models::Int = 10
end

is_classifier(m::H2oModel) = m.isclf

support_multiclass(m::H2oModel) = m.name != "psvm"

function fit!(m::H2oModel, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack name, isclf = m
    ENV["COLS"] = join(columns, '|')
    # init
    max_mem_size = @sprintf("%dG", 0.8 * Sys.total_memory() / 2^30)
    @imports h2o; h2o.init(max_mem_size = max_mem_size)
    @from h2o.estimators.glm imports H2OGeneralizedLinearEstimator
    @from h2o.estimators.psvm imports H2OSupportVectorMachineEstimator
    @from h2o.estimators.random_forest imports H2ORandomForestEstimator
    @from h2o.estimators.gbm imports H2OGradientBoostingEstimator
    @from h2o.estimators.xgboost imports H2OXGBoostEstimator
    @from h2o.estimators.deeplearning imports H2ODeepLearningEstimator
    @from h2o.automl imports H2OAutoML
    h2o.remove_all()
    h2o.api("POST /3/GarbageCollect")
    # create data
    dfx = DataFrame(pymat(x), columns = columns)
    dfy = DataFrame(vec(y), columns = ["label"])
    if isnothing(w)
        hdf = to_h2o(pdhcat(dfx, dfy))
    else
        dfw = DataFrame(vec(w), columns = ["weight"])
        hdf = to_h2o(pdhcat(dfx, dfy, dfw))
    end
    isclf && (hdf["label"] = hdf["label"].asfactor())
    # create model
    pyo = if name == "glm"
        @unpack family, lambda_search, alpha = m
        num_class = isclf ? length(unique(y)) : 1
        family = num_class == 1 ? family : num_class == 2 ? "binomial" : "multinomial"
        H2OGeneralizedLinearEstimator(;@NT(family, lambda_search, alpha)...)
    elseif name == "psvm"
        @unpack gamma, hyper_param = m
        H2OSupportVectorMachineEstimator(gamma = gamma, hyper_param = hyper_param)
    elseif name == "random_forest"
        H2ORandomForestEstimator()
    elseif name == "gbm"
        @unpack ntrees, sample_rate, col_sample_rate, col_sample_rate_per_tree, max_depth, min_rows = m
        H2OGradientBoostingEstimator(;@NT(ntrees, sample_rate, col_sample_rate, col_sample_rate_per_tree, max_depth, min_rows)...)
    elseif name == "xgboost"
        @unpack score_tree_interval, stopping_rounds, ntrees, learn_rate, col_sample_rate = m
        @unpack col_sample_rate_per_tree, max_depth, min_rows, sample_rate = m
        H2OXGBoostEstimator(;@NT(score_tree_interval, stopping_rounds, ntrees, learn_rate,
            col_sample_rate, col_sample_rate_per_tree, max_depth, min_rows, sample_rate)...)
    elseif name == "deeplearning"
        @unpack epochs, adaptive_rate, activation, rho, epsilon, input_dropout_ratio, hidden = m
        H2ODeepLearningEstimator(;@NT(epochs, adaptive_rate, activation, rho, epsilon, input_dropout_ratio)...)
    elseif name == "autoglm"
        @unpack max_models = m
        H2OAutoML(max_models = max_models, max_runtime_secs = 0, include_algos = ["GLM"])
    elseif name == "automl"
        @unpack max_models = m
        H2OAutoML(max_models = max_models, max_runtime_secs = 0, exclude_algos = ["XGBoost"])
    end
    # train
    weights_column = isnothing(w) ? nothing : "weight"
    pyo.train(x = columns, y = "label", weights_column = weights_column, training_frame = hdf)
    if occursin("auto", name)
        pyo, lb = pyo.leader, pyo.leaderboard
        @assert !isnothing(pyo)
        @redirect "summary.txt" begin
            println('*'^100, '\n')
            println("leader summary\n", '='^20)
            @trys println(pyo.summary().as_data_frame().to_string(), '\n')
            println("leaderboard table\n", '='^20)
            @trys println(lb.as_data_frame().to_string(), '\n')
            println("variable importance\n", '='^20)
            @trys println(pyo.varimp(true).to_string(), '\n')
            println('*'^100, '\n')
        end       
    end
    h2o.save_model(pyo, force = true)
    # cleanup
    h2o.remove(hdf)
    h2o.api("POST /3/GarbageCollect")
    @pack! m = pyo
    BSON.bson("model.bson", model = m)
    return m
end

function predict_h2o(m::H2oModel, x)
    @unpack pyo = m
    columns = split(ENV["COLS"], '|')
    df = DataFrame(pymat(x), columns = columns)
    hdf = to_h2o(df)
    index = vec(hdf["index"].as_data_frame().values)
    pred = from_h2o(pyo.predict(hdf)).set_index(index).sort_index()
    @imports h2o
    h2o.remove(hdf)
    h2o.api("POST /3/GarbageCollect")
    return pred
end

predict_proba(m::H2oModel, x) = Array(predict_h2o(m, x).drop(columns = "predict"))

predict(m::H2oModel, x) = Array(predict_h2o(m, x)["predict"])

function gridparams(m::H2oModel)
    glm_comm = Dict("name" => "glm", "lambda_search" => true)
    glm_grid = Dict("alpha" => [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    dl_comm = Dict("name" => "deeplearning", "epochs" => 100, "adaptive_rate" => true,
                    "activation" => "rectifier_with_dropout")
    dl_grid1 = Dict("rho" => [0.9, 0.95, 0.99], "epsilon" => [1e-6, 1e-7, 1e-8, 1e-9],
                    "input_dropout_ratio" => [0.0, 0.05, 0.1, 0.15, 0.2])
    dl_grid2 = Dict("hidden" => [[50], [200], [500]])
    dl_grid3 = Dict("hidden" => [[50, 50], [200, 200], [500, 500]], "hidden_dropout_ratios" =>
                    [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
    gbm_comm = Dict("name" => "gbm", "ntrees" => 100, "sample_rate" => 0.8,
                "col_sample_rate" => 0.8, "col_sample_rate_per_tree" => 0.8)
    gbm_grid = Dict("max_depth" => [6, 7, 8, 10, 15], "min_rows" => [1, 10, 10, 10, 100])
    xgb_comm = Dict("name" => "xgboost", "score_tree_interval" => 5, "stopping_rounds" => 5,
                        "ntrees" => 100, "learn_rate" => 0.05, "col_sample_rate" => 0.8,
                        "col_sample_rate_per_tree" => 0.8)
    xgb_grid = Dict("max_depth" => [10, 20, 5], "min_rows" => [5, 10, 3], "sample_rate" => [0.6, 0.6, 0.8])
    params = []
    append!(params, [merge(glm_comm, p) for p in gridparams(glm_grid, zip)])
    append!(params, [merge(gbm_comm, p) for p in gridparams(gbm_grid, zip)])
    append!(params, [merge(xgb_comm, p) for p in gridparams(xgb_grid, zip)])
    append!(params, [merge(dl_comm, p1, p2) for p1 in gridparams(xgb_grid, zip)
            for p2 in [gridparams(dl_grid2)..., gridparams(dl_grid3)...]])
    return params
end

modelhash(m::H2oModel) = hash(m.pyo)

function from_h2o(hdf)
    @from h2o imports export_file
    dst = tempname() * ".csv"
    for (c, t) in hdf.types
        if occursin(r"string|enum", t)
            hdf[c] = hdf[c].gsub("\"", "|")
        end
    end
    export_file(hdf, dst, force = true)
    df = pd.read_csv(dst)
    rm(dst, force = true)
    for (c, t) in hdf.types
        if t == "time"
            df[c] = pd.to_datetime(df[c], unit = "ms")
        end
    end
    return df
end

function to_h2o(df)
    @imports h2o
    dst = mkpath(joinpath("/dev/shm", tempname()))
    ncpu = h2o.cluster().nodes[1]["nthreads"]
    npart = ceil(Int, length(df) / ncpu)
    df["part"] = repeat(1:ncpu, inner = npart)[1:length(df)]
    df.reset_index(inplace = true)
    if Sys.ARCH == :x86_64
        @imports pyarrow as pa
        @imports pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index = false)
        pq.write_to_dataset(table, root_path = dst, partition_cols = ["part"])
    else
        @imports fastparquet as pq
        pq.write(dst, df, partition_on = ["part"], file_scheme = "hive")
        rm(joinpath(dst, "_metadata"), force = true)
        rm(joinpath(dst, "_common_metadata"), force = true)
    end
    hdf = h2o.import_file(dst)
    rm(dst, force = true, recursive = true)
    return hdf
end

H2oRegressor(;ka...) = H2oModel(;isclf = false, ka...)

H2oClassifier(;ka...) = H2oModel(;isclf = true, ka...)