export TfRanker

using PyCall: python

@with_kw mutable struct TfRanker <: BaseEstimator
    tf::Dict{String, Vector{UInt8}} = Dict()
    nstep::Int = 1000
    lr::Float32 = 1f-2
    pdrop::Float32 = 0
    hidden::String = "10"
    loss::String = "pairwise_logistic_loss"
end

is_classifier(::TfRanker) = false
is_ranker(::TfRanker) = true

function gridparams(m::TfRanker)
    grid = [
        "horizon" => [2, 5, 10, 20, 50],
        "nlevel" => [5, 10, 20],
        "nstep" => [1000, 2000, 5000],
        "lr" => [1f-3, 1f-2, 1f-1],
        "pdrop" => [0f0, 0.3f0, 0.5f0],
        "hidden" => ["10", "20", "50", "100"],
        "loss" =>
        ["pairwise_logistic_loss",
         "pairwise_hinge_loss",
         "pairwise_soft_zero_one_loss",
         "softmax_loss",
         "sigmoid_cross_entropy_loss",
         "mean_squared_loss",
         "list_mle_loss",
         "approx_ndcg_loss"]
    ]
    params = gridparams(grid)
end

function fit!(m::TfRanker, x, y, w = nothing; group = nothing, columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    if isnothing(group) && ndims(x) == 3
        group = repeat([size(x, 2)], size(x, 3))
    end
    ENV["nfea"] = size(x, 1)
    ENV["nlist"] = min(500, minimum(group))
    dst = to_svm(x, y, w, group = group)
    out = mktempdir()
    run(rankcmd(m, dst, out))
    m.tf = readtf(out)
    foreach(rmdir, [dst, out])
    BSON.bson("model.bson", model = m)
    return m
end

function predict(m::TfRanker, x)
    @unpack tf = m
    ENV["nfea"] = size(x, 1)
    ENV["nlist"] = prod(size(x)[2:end])
    dst = to_svm(x)
    out = mktempdir()
    writetf(out, tf)
    run(rankcmd(m, dst, out, predonly = true))
    ŷ = readdlm("preds", Float32)
    foreach(rmdir, ["preds", dst, out])
    return ŷ
end

function rankcmd(m::TfRanker, dst, out; predonly = false)
    @unpack lr, pdrop, hidden, loss, nstep = m
    @unpack nfea, nlist = ENV
    nstep = predonly ? 0 : nstep
    if !usegpu()
        ENV["CUDA_VISIBLE_DEVICES"] = -1
        ENV["MKL_NUM_THREADS"] = 20
        ENV["OPENBLAS_NUM_THREADS"] = 20
        ENV["OMP_NUM_THREADS"] = 20
    end
    tfrank = joinpath(@__DIR__, "tfrank.py")
    `$python $tfrank
    --train_path=$dst
    --vali_path=$dst
    --test_path=$dst
    --output_dir=$out
    --train_batch_size=32
    --num_features=$nfea
    --list_size=$nlist
    --num_train_steps=$nstep
    --learning_rate=$lr
    --dropout_rate=$pdrop
    --hidden_layer_dims=$hidden
    --loss=$loss`
end

readtf(dir) = Dict(src => read(joinpath(dir, src)) for src in readdir(dir))

function writetf(dir, dict)
    mkpath(dir)
    for (src, bytes) in dict
        write(joinpath(dir, src), bytes)
    end
end