__precompile__(true)

module MLSuite

using Statistics, Random, Distributed, Printf, LinearAlgebra, DelimitedFiles, Mmap
using DataStructures, Parameters, ProgressMeter, PyCall, BSON, Requires
using PyCallUtils, PandasLite, HDF5Utils, SVMLightWriter, Iconv
using ScikitLearnBase: BaseEstimator, BaseClassifier, BaseRegressor
import ScikitLearnBase: fit!, predict, predict_proba, is_classifier

reset!(m::BaseEstimator) = nothing
istree(m::BaseEstimator) = false
isrnn(m::BaseEstimator) = false
support_multiclass(m::BaseEstimator) = is_classifier(m)

const VW = joinpath(@__DIR__, "../deps/usr/bin/vw")
const VW_VARINFO = joinpath(@__DIR__, "../deps/usr/bin/vw-varinfo")
const SVM_RANK_LEARN = joinpath(@__DIR__, "../deps/usr/bin/svm_rank_learn")
const SVM_RANK_CLASSIFY = joinpath(@__DIR__, "../deps/usr/bin/svm_rank_classify")
const RANKLIB = joinpath(@__DIR__, "../deps/usr/bin/ranklib")
const LIGHTGBM = joinpath(@__DIR__, "../deps/usr/bin/lightgbm")

include("util.jl")

include("base/vw.jl")
include("base/h2o.jl")
include("base/dai.jl")
include("base/gbm.jl")
include("base/lgbm.jl")

include("class/scikit.jl")

include("regress/grid.jl")
include("regress/scikit.jl")

include("rank/svm.jl")
include("rank/ranklib.jl")
include("rank/tf.jl")

function __init__()
    @require Elemental="902c3f28-d1ec-5e7e-8399-a24c3845ee38" include("regress/el.jl")
end

end