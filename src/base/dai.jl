export DaiModel, DaiRegressor, DaiClassifier

using PyCall: python

mutable struct DaiModel <: BaseEstimator
    key::String
    params::Dict
end

DaiModel(;ka...) = DaiModel("", Dict(String(k) => v for (k, v) in ka))

is_classifier(m::DaiModel) = get(m.params, "is_classification", "True") == "True"

function fit!(m::DaiModel, x, y, w = nothing; eval_set = (), columns = nothing)
    install_h2oai_client()
    columns = something(columns, string.(1:size(x, 1)))
    @unpack key, params = m
    dataset = dump_dai_data(x, y, w, columns = columns)
    testset = dump_dai_data(eval_set..., columns = columns)
    weight_col = isnothing(w) ? "None" : "'weight'"
    is_classification = pop!(params, "is_classification", "True")
    is_time_series = pop!(params, "is_time_series", "False")
    params = join([string(k, "=", v) for (k, v) in params], ",")
    m.key = """
    from h2oai_client import Client
    h2oai = Client(address='http://127.0.0.1:12345', username='username', password='password')
    dataset = h2oai.create_dataset_sync('$dataset')
    if len('$testset') > 0:
        testset = h2oai.create_dataset_sync('$testset')
    else:
        testset = type('dataset', (object,), {'key': None})()
    params = h2oai.get_experiment_tuning_suggestion(
        dataset_key=dataset.key,
        target_col='label',
        is_classification=$is_classification,
        is_time_series=$is_time_series,
        config_overrides=None)
    exp = h2oai.start_experiment_sync(
        dataset_key=dataset.key,
        testset_key=testset.key,
        target_col='label', 
        weight_col=$weight_col,
        is_classification=$is_classification,
        is_time_series=$is_time_series,
        accuracy=params.accuracy, 
        interpretability=params.interpretability,
        time=params.time, $params)
    print(exp.key)
    """ |> pyrun
    rm(dataset, force = true)
    rm(testset, force = true)
    return m
end

function predict_dai(m::DaiModel, x)
    @unpack key = m
    dataset = dump_dai_data(x)
    csv = """
    from h2oai_client import Client
    h2oai = Client(address = "http://127.0.0.1:12345",
            username = "username", password = "password")
    pred = h2oai.make_prediction_sync('$key', '$dataset', false, false)
    csv = h2oai.download(pred.predictions_csv_path, "/tmp")
    print(csv)
    """ |> pyrun
    rm(dataset, force = true)
    pd.read_csv(csv)
end

predict_proba(m::DaiModel, x) = Array(predict_dai(m, x))

function predict(m::DaiModel, x)
    df = predict_dai(m, x)
    is_classifier(m) ? Array(df) :
    Array(df.idxmax(axis = 1))
end

modelhash(m::DaiModel) = m.key

DaiRegressor(;ka...) = DaiModel(is_classification = "True", ka...)

DaiClassifier(;ka...) = DaiModel(is_classification = "False", ka...)

function dump_dai_data(x = nothing, y = nothing, w = nothing; columns = nothing)
    isnothing(x) && return ""
    if isnothing(columns)
        columns = split(ENV["COLS"], '|')
    else
        ENV["COLS"] = join(columns, '|')
    end
    dfx = DataFrame(pymat(x), columns = columns)
    if isnothing(y)
        dfy = DataFrame()
    else
        dfy = DataFrame(vec(y), columns = ["label"])
    end
    if isnothing(w)
        dfw = DataFrame()
    else
        dfw = DataFrame(vec(w), columns = ["weight"])
    end
    df = pdhcat(dfx, dfy, dfw)
    parquet = @sprintf("/dev/shm/%s.parquet", randstring())
    df.to_parquet(parquet)
    return parquet
end

function pyrun(str)
    println(str)
    cmd = Cmd(["python", "-c", str])
    faketime = "/usr/local/lib/faketime/libfaketimeMT.so.1"
    withenv("LD_PRELOAD" => faketime, "FAKETIME" => "-10year") do
        lines = readlines(cmd)
        println(join(lines, '\n'))
        lines[end]
    end
end

function install_h2oai_client()
    if !occursin("h2oai", read(`$python -m pip list`, String))
        whl="http://localhost:12345/static/h2oai_client-1.7.0-py3-none-any.whl"
        run(`$python -m pip install $whl`)
    end
end