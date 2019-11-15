export ScikitClassifier

@with_kw mutable struct ScikitClassifier <: BaseEstimator
    pyo::PyObject = PyNULL()
    name::String = "logistic"
    alpha::Float32 = 1.0
    penalty::String = "l2"
    loss::String = "hinge"
    gamma::Float32 = 0
    n_neighbors::Int = 10
    num_layers::Int = 1
    hidden_size::Int = 100
    lr::Float32 = 0.001
end

is_classifier(::ScikitClassifier) = true

function paramgrid(m::ScikitClassifier)
    grid = Dict()
    grid["logistic"] = OrderedDict(
        "alpha" => [0.001, 0.01, 0.1, 1, 5],
        "penalty" => ["l2", "l1"]
    )
    grid["linearsvc"] = OrderedDict(
        "alpha" => [0.001, 0.01, 0.1, 1, 5],
        "penalty" => ["l2", "l1"],
        "loss" => ["hinge", "squared_hinge"]
    )
    grid["thundersvc"] = OrderedDict(
        "alpha" => [0.001, 0.01, 0.1, 1, 5, 10, 20, 50],
        "gamma" => [0, 0.01, 0.1, 1, 10, 100]
    )
    grid["mlp"] = OrderedDict(
        "num_layers" => [1, 2],
        "hidden_size" => [10, 100],
        "lr" => [0.001, 0.01]
    )
    params = map(["logistic", "linearsvc", "mlp"]) do name
        dict = Dict("name" => [name])
        paramgrid(merge(dict, grid[name]))
    end
    filter(vcat(vec.(params)...)) do d
        !(d["name"] == "linearsvc" &&
          d["penalty"] == "l1" &&
          d["loss"] == "hinge")
    end
end

function fit!(m::ScikitClassifier, x, y, w = nothing; columns = string.(1:size(x, 1)))
    @unpack name, alpha, penalty, loss, gamma = m
    @unpack n_neighbors, num_layers, hidden_size, lr = m
    x, y, w = pymat(x), vec(y), vec(w)
    if name == "logistic"
        @from sklearn.linear_model imports LogisticRegression
        pyo = LogisticRegression(C = 1 / alpha, penalty = penalty, solver = "saga", n_jobs = 20, verbose = true)
    elseif name == "linearsvc"
        @from sklearn.svm imports LinearSVC
        pyo = LinearSVC(dual = loss == "hinge", C = 1 / alpha, penalty = penalty, loss = loss, verbose = true)
    elseif name == "thundersvc"
        @from thundersvm imports SVC
        pyo = SVC(C = 0.5 / alpha, gamma = gamma == 0 ? "auto" : gamma, n_jobs = 20, verbose = true)
    elseif name == "mlp"
        @from sklearn.neural_network imports MLPClassifier
        pyo = MLPClassifier(alpha = alpha, learning_rate_init = lr, verbose = true,
                        hidden_layer_sizes = ntuple(i -> hidden_size, num_layers))
    end
    if name == "thundersvc" || name == "mlp"
        pyo.fit(x, y)
    elseif name == "linearsvc"
        @from sklearn.calibration imports CalibratedClassifierCV
        pyo.fit(x, y, sample_weight = w)
        pyo = CalibratedClassifierCV(pyo, cv = "prefit")
        pyo.fit(x, y, sample_weight = w)
    else
        pyo.fit(x, y, sample_weight = w)
    end
    @pack! m = pyo
    visualize(m, columns)
    return m
end

function predict_proba(m::ScikitClassifier, x)
    @unpack name, pyo = m
    if name == "thundersvc"
        pyo.decision_function(pymat(x))
    else
        pyo.predict_proba(pymat(x))
    end
end

predict(m::ScikitClassifier, x) = m.pyo.predict(pymat(x))

modelhash(m::ScikitClassifier) = hash(m.pyo)

function visualize(m::ScikitClassifier, columns)
    @unpack name, pyo = m
    if name == "logistic"
        write_feaimpt(pyo.coef_, columns)
    elseif name == "linearsvc"
        write_feaimpt(pyo.base_estimator.coef_, columns)
    end
end
