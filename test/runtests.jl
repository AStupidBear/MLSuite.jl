using MLSuite
using Statistics
using Pkg
using Test

cd(mktempdir())

F, N, T = 10, 100, 10
x = randn(Float32, F, N, T)
w = rand(Float32, N, T)
y = mean(x, dims = 1)

regressors = [
    GridRegressor(),
    VWRegressor(),
    ScikitRegressor(name = "ridge", alpha = 0.01),
    ScikitRegressor(name = "lasso", alpha = 0.01),
    ScikitRegressor(name = "ridgecv"),
    ScikitRegressor(name = "lassocv"),
    ScikitRegressor(name = "kernelridge"),
    ScikitRegressor(name = "knn"),
    ScikitRegressor(name = "linearsvr"),
    ScikitRegressor(name = "svr"),
    ScikitRegressor(name = "mlp"),
    ScikitRegressor(name = "lightgbm"),
    H2oRegressor(name = "glm"),
    H2oRegressor(name = "random_forest"),
    H2oRegressor(name = "gbm"),
    H2oRegressor(name = "xgboost"),
    H2oRegressor(name = "deeplearning"),
    H2oRegressor(name = "automl"),
    H2oRegressor(name = "autoglm"),
    GbmRegressor(name = "lightgbm"),
    GbmRegressor(name = "xgboost"),
    GbmRegressor(name = "catboost"),
    LgbmRegressor(warm_start = false),
    LgbmRegressor(warm_start = true)
]

if "Elemental" ∈ keys(Pkg.installed())
    using Elemental
    push!(regressors, ElRegressor(name = "lstsq"))
    push!(regressors, ElRegressor(name = "ridge", alpha = 0.1))
end

for model in regressors
    print(model)
    MLSuite.fit!(model, x, y, w)
    ŷ = MLSuite.predict(model, x)
    r2 = r2_score(vec(ŷ), vec(y))
    @test r2 > 0.5
end

classifiers = [
    VWClassifier(),
    ScikitClassifier(name = "logistic"),
    ScikitClassifier(name = "linearsvc"),
    ScikitClassifier(name = "svc"),
    ScikitClassifier(name = "mlp"),
    H2oClassifier(name = "glm"),
    H2oClassifier(name = "psvm"),
    H2oClassifier(name = "random_forest"),
    H2oClassifier(name = "gbm"),
    H2oClassifier(name = "xgboost"),
    H2oClassifier(name = "deeplearning"),
    H2oClassifier(name = "autoglm"),
    H2oClassifier(name = "automl"),
    GbmClassifier(name = "lightgbm"),
    GbmClassifier(name = "xgboost"),
    GbmClassifier(name = "catboost"),
    LgbmClassifier()
]

for binary in [true, false], model in classifiers
    print(model)
    !binary && !MLSuite.support_multiclass(model) && continue
    y′ = binary ? signone.(y) : @. ifelse(abs(y) > 0.1, sign(y) + 1.0, 1.0)
    MLSuite.fit!(model, x, y′, w)
    ŷ = MLSuite.predict(model, x)
    prob = MLSuite.predict_proba(model, x)
    acc = accuracy_score(vec(ŷ), vec(y′))
    @assert acc > 0.6
end

rankers = [
    SvmRanker(),
    TfRanker(loss = "pairwise_hinge_loss"),
    TfRanker(loss = "pairwise_soft_zero_one_loss"),
    TfRanker(loss = "softmax_loss"),
    TfRanker(loss = "sigmoid_cross_entropy_loss"),
    TfRanker(loss = "mean_squared_loss"),
    TfRanker(loss = "list_mle_loss"),
    TfRanker(loss = "approx_ndcg_loss"),
    RanklibRanker(ranker = 0),
    RanklibRanker(ranker = 1),
    RanklibRanker(ranker = 2),
    RanklibRanker(ranker = 3, metric2t = "NDCG@50"),
    RanklibRanker(ranker = 3, metric2t = "DCG@50"),
    RanklibRanker(ranker = 3, metric2t = "P@50"),
    RanklibRanker(ranker = 3, metric2t = "RR@50"),
    RanklibRanker(ranker = 3, metric2t = "ERR@50"),
    RanklibRanker(ranker = 4, metric2t = "NDCG@50"),
    RanklibRanker(ranker = 4, metric2t = "DCG@50"),
    RanklibRanker(ranker = 4, metric2t = "P@50"),
    RanklibRanker(ranker = 4, metric2t = "RR@50"),
    RanklibRanker(ranker = 4, metric2t = "ERR@50"),
    RanklibRanker(ranker = 6, metric2t = "NDCG@50"),
    RanklibRanker(ranker = 6, metric2t = "DCG@50"),
    RanklibRanker(ranker = 6, metric2t = "P@50"),
    RanklibRanker(ranker = 6, metric2t = "RR@50"),
    RanklibRanker(ranker = 6, metric2t = "ERR@50"),
    RanklibRanker(ranker = 7),
    RanklibRanker(ranker = 8),
    GbmRanker(name = "lightgbm", objective = "rank:ndcg"),
    GbmRanker(name = "xgboost", objective = "rank:pairwise"),
    GbmRanker(name = "catboost", objective = "rank:pairlogit")
]

for model in rankers
    print(model)
    y′ = @. ifelse(y > 0, 1, 0)
    MLSuite.fit!(model, x, y′, w)
    ŷ = MLSuite.predict(model, x)
    ŷ .-= mean(ŷ)
    ŷ = @. ifelse(ŷ > 0, 1, 0)
    acc = accuracy_score(vec(ŷ), vec(y′))
end