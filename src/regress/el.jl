using Elemental, MPI
using MPIClusterManagers
const MCM = MPIClusterManagers

export ElRegressor, @mb_mpi_do

macro mb_mpi_do(mngr, ex)
    esc(:(if isa($mngr, MCM.MPIManager)
    MCM.@mpi_do($mngr, $ex)
    else
        @eval Main $ex
    end))
end

@with_kw mutable struct ElRegressor <: BaseEstimator
    W::Vector{Float32} = zeros(Float32, 0)
    name::String = "lstsq"
    alpha::Float32 = 0f0
end

is_classifier(::ElRegressor) = false

function gridparams(m::ElRegressor)
    grid = [
        "name" => ["lstsq", "ridge", "lasso"],
        "alpha" => [0.001, 0.01, 0.1, 1, 10],
    ]
    filter(gridparams(grid)) do d
        d["name"] != "lstsq" || d["alpha"] == 0.001
    end
end

function fit!(m::ElRegressor, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack name, alpha = m
    mngr = try Main.mngr catch end
    yr = reshape(y, 1, :)
    xr = reshape(x, size(x, 1), :)
    @eval Main xr, yr = $xr, $yr
    @mb_mpi_do mngr using TimerOutputs
    @mb_mpi_do mngr begin
        using Elemental, Distributed, SparseArrays, LinearAlgebra
        reset_timer!()
        if myid() != 1
            xr = spzeros(Float32, $(size(xr))...)
            yr = spzeros(Float32, $(size(yr))...)
        end
        B = Elemental.DistMatrix(Float32)
        X = Elemental.DistMatrix(Float32)
        @timeit "copy B" copy!(B, permutedims(yr))
        if !@isdefined(A) || size(A, 1) != size(B, 1)
            A = Elemental.DistMatrix(Float32)
            myid() == 1 && @info("copying to A...")
            @timeit "copy A" copy!(A, transpose(xr))
            Δ = Elemental.DistMatrix(Float32)
            Elemental.gaussian!(Δ, size(A)...)
            @timeit "A += 0.01Δ" BLAS.axpy!(5f-3, Δ, A)
        end
        if $name == "lstsq"
            Elemental.leastSquares!(A, B, X)
        elseif $name == "ridge"
            Elemental.ridge!(A, B, $alpha, X)
        elseif $name == "lasso"
            Elemental.bpdn!(A, B, $alpha, X)
        end
        X_array = Array(X)
        print_timer()
    end
    m.W = vec(Main.X_array)
    BSON.bson("model.bson", model = m)
    visualize(m, columns)
    return m
end

predict(m::ElRegressor, x) = vec(transpose(m.W) * reshape(x, size(x, 1), :))

visualize(m::ElRegressor, columns) = write_feaimpt(m.W, columns)

modelhash(m::ElRegressor) = hash(m.W)