export ScikitRegressor

@with_kw mutable struct ScikitRegressor <: BaseEstimator
    pyo::PyObject = PyNULL()
    name::String = "lightgbm"
    warm_start::Bool = false
    alpha::Float32 = 1.0
    fit_intercept::Bool = true
    kernel::String = "rbf"
    gamma::Float32 = 0
    n_neighbors::Int = 10
    num_layers::Int = 1
    hidden_size::Int = 100
    lr::Float32 = 0.001
    n_estimators::Int = 20
    num_leaves::Int = 31
    colsample_bytree::Float32 = 1.0
    subsample::Float32 = 1.0
    reg_lambda::Float32 = 0.1
    min_child_samples::Int = 100
end

is_classifier(::ScikitRegressor) = false

function paramgrid(m::ScikitRegressor)
    grid = Dict()
    grid["lasso"] = grid["ridge"] = OrderedDict("alpha" => [0.001, 0.01, 0.1, 1])
    grid["kernelridge"] = OrderedDict(
        "alpha" => [0.001, 0.01, 0.1, 1],
        "kernel" => ["rbf", "poly", "sigmoid"],
        "gamma" => [0, 0.01, 0.1, 1, 10, 100]
    )
    grid["knn"] = OrderedDict("n_neighbors" => [10, 100, 1000])
    grid["linearsvr"] = grid["ridge"]
    grid["svr"] = grid["kernelridge"]
    grid["mlp"] = OrderedDict(
        "num_layers" => [1, 2],
        "hidden_size" => [10, 100],
        "lr" => [0.001, 0.01]
    )
    grid["lightgbm"] = OrderedDict(
        "num_leaves" => [15, 31],
        "min_child_samples" => [1000, 100, 5000],
        "n_estimators" => [20, 50],
        "reg_lambda" => [0.1, 1, 10],
        "subsample" => [1, 0.8],
        "colsample_bytree" => [1, 0.8],
    )
    params = map(["lightgbm", "ridge", "mlp", "svr"]) do name
        dict = Dict("name" => [name])
        paramgrid(merge(dict, grid[name]))
    end
    params = vcat(vec.(params)...)
end

function fit!(m::ScikitRegressor, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    x, y, w = pymat(x), vec(y), vec(w)
    @unpack name, warm_start, alpha, fit_intercept, kernel, gamma, n_neighbors = m
    @unpack num_layers, hidden_size, lr, n_estimators, num_leaves = m
    @unpack colsample_bytree, subsample, reg_lambda, min_child_samples = m
    if name == "ridge"
        @from sklearn.linear_model imports Ridge
        pyo = Ridge(alpha = alpha)
    elseif name == "lasso"
        @from sklearn.linear_model imports Lasso
        pyo = Lasso(alpha = alpha)
    elseif name == "ridgecv"
        @from sklearn.linear_model imports RidgeCV
        pyo = RidgeCV(fit_intercept = fit_intercept)
    elseif name == "lassocv"
        @from sklearn.linear_model imports LassoCV
        pyo = LassoCV(fit_intercept = fit_intercept)
    elseif name == "kernelridge"
        @from sklearn.kernel_ridge imports KernelRidge
        pyo = KernelRidge(alpha = alpha, kernel = kernel, gamma = gamma == 0 ? nothing : gamma)
    elseif name == "knn"
        @from sklearn.neighbors imports KNeighborsRegressor
        pyo = KNeighborsRegressor(n_neighbors = n_neighbors)
    elseif name == "linearsvr"
        @from sklearn.svm imports LinearSVR
        pyo = LinearSVR(dual = false, C = 0.5 / alpha, loss = "squared_epsilon_insensitive")
    elseif name == "svr"
        use_thunder = true
        try
            @from thundersvm imports SVR
            pyo = SVR(kernel = kernel, C = 0.5 / alpha, gamma = gamma == 0 ? "auto" : gamma, n_jobs = 20)
        catch e
            use_thunder = false
            @from sklearn.svm imports SVR
            pyo = SVR(kernel = kernel, C = 0.5 / alpha, gamma = gamma == 0 ? "auto" : gamma)
        end
    elseif name == "mlp"
        @from sklearn.neural_network imports MLPRegressor
        pyo = MLPRegressor(
            hidden_layer_sizes = ntuple(i -> hidden_size, num_layers),
            alpha = alpha,
            learning_rate_init = lr,
            verbose = true,
            warm_start = true,
        )
        if warm_start && occursin("MLP", string(m.pyo))
            @unpack pyo = m
        end
    elseif name == "lightgbm"
        @from lightgbm imports LGBMRegressor
        if warm_start
            @imports lightgbm
            @from unidecode imports unidecode
            dset = lightgbm.Dataset(data = x, label = y, weight = w)
            columns′ = [filter(!isspace, unidecode(c)) for c in columns]
            dset.set_feature_name(columns′)
            params = Dict(pairs((n_jobs = 20, max_bin = 15, num_leaves = num_leaves,
                        subsample = subsample, colsample_bytree = colsample_bytree,
                        reg_lambda = reg_lambda, min_child_samples = min_child_samples)))
            init_model = isfile("lightgbm") ? "lightgbm" : nothing
            pyo = lightgbm.train(params, dset, num_boost_round = n_estimators, init_model = init_model)
            pyo.save_model("lightgbm")
        else
            pyo = LGBMRegressor(n_jobs = 20, max_bin = 15, num_leaves = num_leaves,
                        subsample = subsample, colsample_bytree = colsample_bytree,
                        reg_lambda = reg_lambda, min_child_samples = min_child_samples)
            pyo.fit(x, y, sample_weight = w)
        end
    elseif name == "autosk"
        @from autosklearn.regression imports AutoSklearnRegressor
        AutoSklearnRegressor(resampling_strategy = "cv", resampling_strategy_arguments = Dict("folds" => 5))
    end
    if name == "svr" && use_thunder || occursin(r"mlp|lasso|knn", name)
        pyo.fit(x, y)
    elseif name != "lightgbm"
        pyo.fit(x, y, sample_weight = w)
    end
    @pack! m = pyo
    visualize(m, columns)
    return m
end

predict(m::ScikitRegressor, x) = m.pyo.predict(pymat(x))

istree(m::ScikitRegressor) = occursin(r"tree|gbm"i, m.name)

reset!(m::ScikitRegressor) = rm("lightgbm", force = true)

modelhash(m::ScikitRegressor) = hash(m.pyo)

function visualize(m::ScikitRegressor, columns)
    @unpack name, pyo = m
    if name == "lightgbm"
        gbm = GbmModel(pyo = pyo)
        visualize(gbm, columns)
    elseif occursin(r"^ridge|lasso", name)
        write_feaimpt(pyo.coef_, columns)
    end
end