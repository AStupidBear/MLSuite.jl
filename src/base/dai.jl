export DaiModel, DaiRegressor, DaiClassifier

@with_kw mutable struct DaiModel <: BaseEstimator
    h2oai::PyObject = PyNULL()
    exp::PyObject = PyNULL()
    isclf::Bool = true
    ists::Bool = false
end

is_classifier(m::DaiModel) = m.clf

function fit!(m::DaiModel, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    ENV["COLS"] = join(columns, '|')
    @unpack isclf, ists = m
    @from h2oai_client imports Client
    h2oai = Client(address = "http://127.0.0.1:12345",
                username = "abcd", password = "dcba")
    dfx = DataFrame(pymat(x), columns = columns)
    dfy = DataFrame(vec(y), columns = ["label"])
    if isnothing(w)
        df  = pdhcat(dfx, dfy)
    else
        dfw = DataFrame(vec(w), columns = ["weight"])
        df  = pdhcat(dfx, dfy, dfw)
    end
    parquet = "/dev/shm/dai.parquet"
    pd.to_parquet(df, parquet)
    dset = h2oai.create_dataset_sync(parquet)
    weight_col = isnothing(w) ? nothing : "weight"
    params = h2oai.get_experiment_tuning_suggestion(
        dataset_key = dset.key, target_col = "label",
        weight_col = weight_col, is_classification = isclf,
        is_time_series = ists, config_overrides = nothing).dump()
    println(params)
    params = [Symbol(k) => v for (k, v) in params]
    exp = h2oai.start_experiment_sync(;params...)
    rm(parquet, force = true)
    @pack! m = h2oai, exp
    return m
end

function predict_dai(m::DaiModel, x)
    @unpack h2oai, exp = m
    columns = split(ENV["COLS"], '|')
    df = DataFrame(pymat(x), columns = columns)
    parquet = "/dev/shm/dai.parquet"
    pd.to_parquet(df, parquet)
    pred = h2oai.make_prediction_sync(exp.key, parquet, false, false)
    rm(parquet, force = true)
    csv = h2oai.download(pred.predictions_csv_path, "/tmp")
    pd.read_csv(csv)
end

predict_proba(m::DaiModel, x) = Array(predict_dai(m, x))

function predict(m::DaiModel, x)
    df = predict_dai(m, x)
    !m.isclf ? Array(df) :
    Array(df.idxmax(axis = 1))
end

modelhash(m::DaiModel) = hash(m.pyo)

DaiRegressor(;ka...) = DaiModel(;isclf = false, ka...)

DaiClassifier(;ka...) = DaiModel(;isclf = true, ka...)