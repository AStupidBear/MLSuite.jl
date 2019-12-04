__precompile__(true)

module MLSuite

using Statistics, Random, Distributed, Printf, LinearAlgebra, DelimitedFiles, Mmap
using DataStructures, Parameters, ProgressMeter, PyCall, BSON, Requires
using PyCallUtils, PandasLite, HDF5Utils, SVMLightWriter, Iconv
using ScikitLearnBase: BaseEstimator, BaseClassifier, BaseRegressor
import ScikitLearnBase: fit!, predict, predict_proba, is_classifier
import MLSuiteBase: paramgrid, reset!, istree, isrnn, is_ranker, support_multiclass, modelhash

const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())
const VW = joinpath(DEPOT_PATH[1], "mlsuite/bin/vw")
const VW_VARINFO = joinpath(DEPOT_PATH[1], "mlsuite/bin/vw-varinfo")
const SVM_RANK_LEARN = joinpath(DEPOT_PATH[1], "mlsuite/bin/svm_rank_learn")
const SVM_RANK_CLASSIFY = joinpath(DEPOT_PATH[1], "mlsuite/bin/svm_rank_classify")
const RANKLIB = joinpath(DEPOT_PATH[1], "mlsuite/bin/ranklib")
const LIGHTGBM = joinpath(DEPOT_PATH[1], "mlsuite/bin/lightgbm")

export accuracy_score, r2_score
accuracy_score(a...; ka...) = pyimport("sklearn.metrics").accuracy_score(a...; ka...)
r2_score(a...; ka...) = pyimport("sklearn.metrics").r2_score(a...; ka...)

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
    boostlib = joinpath(DEPOT_PATH[1], "mlsuite/boost/lib") 
    ENV["LD_LIBRARY_PATH"] = boostlib * ":" * get(ENV, "LD_LIBRARY_PATH", "")
    @require Elemental="902c3f28-d1ec-5e7e-8399-a24c3845ee38" include("regress/el.jl")
end

end