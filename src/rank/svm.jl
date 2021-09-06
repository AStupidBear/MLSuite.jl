export SvmRanker

@with_kw mutable struct SvmRanker <: BaseEstimator
    svm::Vector{UInt8} = UInt8[]
    c::Float32 = 0.01
    p::Int = 1
    o::Int = 2
end

is_classifier(::SvmRanker) = false
is_ranker(::SvmRanker) = true

function gridparams(m::SvmRanker)
    grid = ["c" => [0.01, 0.1, 1, 10, 100]]
    params = gridparams(grid)
end

function fit!(m::SvmRanker, x, y, w = nothing; group = nothing, columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    if isnothing(group) && ndims(x) == 3
        group = repeat([size(x, 2)], size(x, 3))
    end
    @unpack c, p, o = m
    dst = to_svm(x, y, w, group = group)
    run(`$SVM_RANK_LEARN -c $c -p $p -o $o $dst svm`)
    m.svm = read("svm")
    foreach(rm, ["svm", dst])
    BSON.bson("model.bson", model = m)
    return m
end

function predict(m::SvmRanker, x)
    @unpack svm = m
    dst = to_svm(x)
    write("svm", svm)
    run(`$SVM_RANK_CLASSIFY $dst svm preds`)
    ŷ = readdlm("preds", Float32)
    foreach(rm, ["svm", "preds", dst])
    return ŷ
end