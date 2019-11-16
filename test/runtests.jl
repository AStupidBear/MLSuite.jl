using MLSuite
using PyCall
using PyCallUtils
using Statistics
using Test

cd(mktempdir())

@from sklearn.metrics imports accuracy_score, r2_score

F, N, T = 10, 1000, 10
x = randn(Float32, F, N, T)
w = rand(Float32, N, T)
y = mean(x, dims = 1)

regressors = [
    # GridRegressor(),
    VWRegressor(),
    ScikitRegressor(name = "ridge", alpha = 0),
    ScikitRegressor(name = "lasso", alpha = 0),
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
    LgbmClassifier(),
]

for binary in [true, false], model in classifiers
    print(model)
    !binary && !MLSuite.support_multiclass(model) && continue
    y′ = binary ? signone.(y) : @. ifelse(abs(y) > 0.1, sign(y) + 1.0, 1.0)
    MLSuite.fit!(model, x, y′, w)
    ŷ = MLSuite.predict(model, x)
    prob = MLSuite.predict_proba(model, x)
    acc = accuracy_score(vec(ŷ), vec(y′))
    @assert acc > 0.7
end