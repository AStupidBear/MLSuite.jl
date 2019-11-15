export RanklibRanker

@with_kw mutable struct RanklibRanker <: BaseEstimator
    ranklib::Vector{UInt8} = UInt8[]
    ranker::Int = 0
    metric2t::String = "MAP"
end

is_classifier(::RanklibRanker) = false

function paramgrid(m::RanklibRanker)
    grid = OrderedDict(
        "ranker" => 0:8,
        "metric2t" =>
        ["MAP"; ["$s@$k" for k in [20, 50, 100]
        for s in split("NDCG DCG P RR ERR")]]
    )
    filter(paramgrid(grid)) do d
        d["metric2t"] == "MAP" || d["ranker"] ∈ [3, 4, 6]
    end
end

function fit!(m::RanklibRanker, x, y, w = nothing; group = nothing, columns = string.(1:size(x, 1)))
    @unpack ranker, metric2t = m
    dst = to_svm(x, y, w, group = group)
    run(`$RANKLIB -train $dst -save ranklib -ranker $ranker -metric2t $metric2t`)
    m.ranklib = read("ranklib")
    foreach(rm, ["ranklib", dst])
    return m
end

function predict(m::RanklibRanker, x)
    @unpack ranklib = m
    dst = to_svm(x)
    write("ranklib", ranklib)
    run(`$RANKLIB -load ranklib -rank $dst -score preds`)
    ŷ = readdlm("preds", Float32)[:, 3]
    foreach(rm, ["ranklib", "preds", dst])
end