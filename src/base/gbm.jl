export GbmModel, GbmRegressor, GbmClassifier, GbmRanker

@with_kw mutable struct GbmModel <: BaseEstimator
    pyo::PyObject = PyNULL()
    name::String = "lightgbm"
    max_bin::Int = 16
    objective::String = "binary:logistic"
    max_depth::Int = 5
    min_child_weight::Float32 = 1
    gamma::Float32 = 0
    num_leaves::Int = 31
    min_child_samples::Int = 1000
    max_position::Int = 200
    label_gain::String = join(0:30, ',')
    n_estimators::Int = 50
    reg_lambda::Float32 = 0.1
    subsample::Float32 = 1
    colsample_bytree::Float32 = 1
    colsample_bylevel::Float32 = 1
    bagging_temperature::Float32 = 1
end

is_classifier(m::GbmModel) = occursin(r"binary|multi", m.objective)
is_ranker(m::GbmModel) = occursin("rank", m.objective)

function gridparams(m::GbmModel)
    @unpack objective, name = m
    rank = is_ranker(m)
    grid = [
        "n_estimators" => [20, 50, 100],
        "reg_lambda" => [0.1, 1, 10],
        [   
            "name" => "lightgbm",
            "objective" => rank ? ["rank:ndcg"] : [objective],
            "num_leaves" => [15, 31],
            "max_position" => rank ? [100, 200, 500, 1000] : [1],
            "label_gain" => rank ? [join(0:30, ','), ""] : [""],
            "min_child_samples" => [1000, 100, 5000],
            "subsample" => [1, 0.8],
            "colsample_bytree" => [1, 0.8]
        ],
        [
            "name" => ["xgboost"],
            "objective" => rank ? ["rank:pairwise", "rank:map", "rank:ndcg"] : [objective],
            "max_depth" => [5, 3, 7],
            "min_child_weight" => [1, 10, 100],
            "gamma" => [0, 0.01, 0.1, 1],
            "subsample" => [1, 0.8],
            "colsample_bytree" => [1, 0.8],
        ],
        [
            "name" => ["catboost"],
            "objective" => !rank ? [objective] :
                ["rank:queryrmse", "rank:yetirank", 
                "rank:yetirankpairwise", "rank:pairlogit", 
                "rank:pairlogitpairwise"],
            "max_depth" => [5, 3, 7],
            "min_child_samples" => !usegpu() ? [1000] : [1000, 100, 5000],
            "bagging_temperature" => [0, 1, 2, 5, 10],
            "colsample_bylevel" => usegpu() ? [1] : [1, 0.8],
        ]
    ]
    return griparams(grid)
end

function fit!(m::GbmModel, x, y, w = nothing; group = nothing, columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    if is_ranker(m) && isnothing(group) && ndims(x) == 3
        group = repeat([size(x, 2)], size(x, 3))
    end
    x, y, w, group = pymat(x), vec(y), vec(w), vec(group)
    x = DataFrame(x, columns = columns)
    @unpack name, max_bin, objective, max_depth, min_child_weight, gamma = m
    @unpack num_leaves, min_child_samples, max_position, label_gain = m
    @unpack max_depth, n_estimators, reg_lambda, subsample, colsample_bytree = m
    @unpack colsample_bylevel, bagging_temperature = m
    num_class = occursin("reg", objective) ? 1 : length(unique(y))
    objective = occursin("binary", objective) && num_class > 2 ? "multi:softmax" : objective
    num_class = occursin("multi", objective) ? num_class : 1
    y = occursin("binary", objective) && name == "xgboost" ? (y .+ 1) ./ 2 : y
    objective = denormalize_objective(objective, name)
    min_child_samples = min(length(y) ÷ 10, min_child_samples)
    if name == "xgboost"
        @from xgboost imports XGBModel, XGBRanker
        XGBModel = !is_ranker(m) ? XGBModel : XGBRanker
        tree_method = usegpu() && is_ranker(m) ? "gpu_hist" : "auto"
        pyo = XGBModel(n_jobs = 20, max_bin = max_bin, tree_method = tree_method, single_precision_histogram = true, n_gpus = countgpus(), num_class = num_class)
        pyo.set_params(;@NT(objective, max_depth, min_child_weight, n_estimators, reg_lambda, subsample, gamma)...)
        tree_method == "auto" && pyo.set_params(colsample_bytree = colsample_bytree)
        if isnothing(group)
            @time pyo.fit(x, y, sample_weight = w, eval_set = [(x, y)], sample_weight_eval_set = bra(w), verbose = true)
        else
            if length(group) != length(w)
                @imports numpy as np
                w = map(sum, np.split(w, cumsum(group[1:end - 1])))
                @show length(w), length(group)
            end
            @time pyo.fit(x, y, group, sample_weight = w, eval_set = [(x, y)], sample_weight_eval_set = bra(w), eval_group = bra(group), verbose = true)
        end
    elseif name == "lightgbm"
        @from lightgbm imports LGBMModel
        @from unidecode imports unidecode
        device = usegpu() ? "gpu" : "cpu"
        pyo = LGBMModel(n_jobs = 20, max_bin = max_bin - 1, device = device, label_gain = label_gain, num_class = num_class)
        pyo.set_params(;@NT(objective, num_leaves, min_child_samples, max_position, n_estimators, reg_lambda, subsample, colsample_bytree)...)
        x.columns = columns′ = [replace(filter(!isspace, unidecode(c)), ":" => "-") for c in columns]
        @time pyo.fit(x, y, w, group = group, eval_set = [(x, y)], eval_sample_weight = bra(w), eval_group = bra(group), feature_name = columns′)
    elseif name == "catboost"
        @from catboost imports CatBoost
        @imports numpy as np
        min_child_samples = usegpu() ? min_child_samples : 1
        colsample_bylevel = usegpu() ? 1 : colsample_bylevel
        task_type = usegpu() ? "GPU" : "CPU"
        devices = string("0:", countgpus() - 1)
        loss_function = objective
        pyo = CatBoost(pairs((thread_count = 20, max_bin = max_bin - 1, task_type = task_type, devices = devices, learning_rate = 0.1)))
        pyo.set_params(;@NT(loss_function, max_depth, n_estimators, reg_lambda, colsample_bylevel, min_child_samples, bagging_temperature)...)
        group = isnothing(group) ? nothing : np.repeat(axes(group, 1), group)
        @time pyo.fit(x, y, sample_weight = w, group_id = group)
    end
    @pack! m = pyo
    BSON.bson("model.bson", model = m)
    visualize(m, columns)
    return m
end

function predict_gbm(m::GbmModel, x)
    @unpack name, pyo = m
    if name == "xgboost"
        pyo.predict(pymat(x), output_margin = true, validate_features = false)
    elseif name == "lightgbm"
        pyo.predict(pymat(x), raw_score = true)
    else
        pyo.predict(pymat(x))
    end
end

function predict_proba(m::GbmModel, x)
    ŷ = predict_gbm(m, x)
    if ndims(ŷ) == 1
        ŷ = sigmoid.(ŷ)
        hcat(1 .- ŷ,  ŷ)
    else
        softmax(ŷ, dims = 2)
    end
end

function predict(m::GbmModel, x)
    ŷ = predict_gbm(m, x)
    !is_classifier(m) && return ŷ
    if ndims(ŷ) == 1
        signone.(ŷ)
    else
        [I[2] - 1 for I in argmax(ŷ, dims = 2)]
    end
end

istree(m::GbmModel) = true

function visualize(m::GbmModel, columns)
    @unpack pyo = m
    mname, tname = pymodulename(pyo), pytypename(pyo)
    plot_tree = parseenv("PLOT_TREE", false)
    graphs, w = [], []
    if occursin("lightgbm", mname)
        bst = tname == "Booster" ? pyo : pyo.booster_
        w = bst.feature_importance()
        @from lightgbm imports create_tree_digraph
        plot_tree && for n in 1:min(5, bst.num_trees())
            push!(graphs, create_tree_digraph(bst, tree_index = n - 1))
        end
    elseif occursin("xgboost", mname)
        bst = tname == "Booster" ? pyo : pyo.get_booster()
        score = bst.get_score()
        columns = collect(keys(score))
        w = collect(values(score))
        @from xgboost imports to_graphviz
        plot_tree && for n in 1:min(5, bst.best_iteration)
            push!(graphs, to_graphviz(bst, num_trees = n - 1))
        end
    else
        return
    end
    write_feaimpt(w, columns)
    plot_tree && for (n, g) in enumerate(graphs)
        try g.render("gbm/$n.pdf", cleanup = "true") catch e; println(e) end
    end
    plot_tree && try bash("pdfcat -o gbm.pdf gbm/*.pdf && rm -rf gbm") catch end
end

function save(m::GbmModel, dst = "model.gbm")
    @unpack pyo = m
    mname, tname = pymodulename(pyo), pytypename(pyo)
    if occursin("lightgbm", mname)
        bst = tname == "Booster" ? pyo : pyo.booster_
    elseif occursin("xgboost", mname)
        bst = tname == "Booster" ? pyo : pyo.get_booster()
    elseif occursin("catboost", mname)
        bst = pyo
    else
        return
    end
    bst.save_model(dst)
end

function denormalize_objective(objective, name)
    mapping = Dict(v => k for (k, v) in objective_mapping[name])
    get(mapping, objective, objective)
end

function normalize_objective(objective)
    mapping = merge(values(objective_mapping)...)
    get(mapping, objective, objective)
end

const objective_mapping = Dict(
    "xgboost" => Dict(
        "reg:squarederror" => "reg:mse",
        "reg:squaredlogerror" => "reg:msle",
        "reg:gamma" => "reg:gamma",
        "reg:tweedie" => "reg:tweedie",
        "reg:logistic" => "reg:logistic",
        "binary:logistic" => "binary:logistic",
        "binary:logitraw" => "binary:logitraw",
        "binary:hinge" => "binary:hinge",
        "count:poisson" => "reg:poisson",
        "survival:cox" => "reg:cox",
        "multi:softmax" => "multi:softmax",
        "multi:softprob" => "multi:softprob",
        "rank:pairwise" => "rank:pairwise",
        "rank:ndcg" => "rank:ndcg",
        "rank:map" => "rank:map"
    ),
    "lightgbm" => Dict(
        "regression" => "reg:mse",
        "regression_l1" => "reg:mae",
        "huber" => "reg:huber",
        "fair" => "reg:fair",
        "poisson" => "reg:poisson",
        "quantile" => "reg:quantile",
        "mape" => "reg:mape",
        "gamma" => "reg:gamma",
        "tweedie" => "reg:tweedie",
        "binary" => "binary:logistic",
        "cross_entropy" => "binary:cross_entropy",
        "cross_entropy_lambda" => "binary:cross_entropy_lambda",
        "multiclass" => "multi:softmax",
        "multiclassova" => "multi:ova",
        "lambdarank" => "rank:ndcg"
    ),
    "catboost" => Dict(
        "MAE" => "reg:mae",
        "MAPE" => "reg:mape",
        "Poisson" => "reg:poisson",
        "Quantile" => "reg:quantile",
        "RMSE" => "reg:mse",
        "LogLinQuantile" => "reg:loglinquantile",
        "Huber" => "reg:huber",
        "Expectile" => "reg:expectile",
        "Logloss" => "binary:logistic",
        "CrossEntropy" => "binary:crossentropy",
        "MultiClass" => "multi:softmax",
        "MultiClassOneVsAll" => "multi:ova",
        "PairLogit" => "rank:pairlogit",
        "PairLogitPairwise" => "rank:pairlogitpairwise",
        "YetiRank" => "rank:yetirank",
        "YetiRankPairwise" => "rank:yetirankpairwise",
        "QueryCrossEntropy" => "rank:querycrossentropy",
        "QueryRMSE" => "rank:queryrmse",
        "QuerySoftMax" => "rank:querysoftmax"
    )
)

GbmRegressor(;ka...) = GbmModel(;objective = "reg:mse", ka...)

GbmClassifier(;ka...) = GbmModel(;objective = "binary:logistic", ka...)

GbmRanker(;ka...) = GbmModel(;objective = "rank:ndcg", ka...)