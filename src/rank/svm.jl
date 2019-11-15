export SvmRanker

@with_kw mutable struct SvmRanker <: BaseEstimator
    svm::Vector{UInt8} = UInt8[]
    c::Float32 = 0.01
    p::Int = 1
    o::Int = 2
end

is_classifier(::SvmRanker) = false

function paramgrid(m::SvmRanker)
    grid = OrderedDict(
        "c" => [0.01, 0.1, 1, 10, 100],
        "p" => [1, 2],
        "o" => [2, 1]
    )
    params = paramgrid(grid)
end

function fit!(m::SvmRanker, x, y, w = nothing; group = nothing, columns = string.(1:size(x, 1)))
    @unpack c, p, o = m
    dst = to_svm(x, y, w, group = group)
    run(`$SVM_RANK_LEARN -c $c -p $p -o $o $dst svm`)
    m.svm = read("svm")
    foreach(rm, ["svm", dst])
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