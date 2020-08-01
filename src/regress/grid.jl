export GridRegressor

const NG = 16 # number of grids

@with_kw mutable struct GridRegressor <: BaseEstimator
    W::Array{Float32} = zeros(Float32, 0)
    α::Float32 = 1e-4
    λ::Float32 = 0
    max_iter::Int = 1000
    tol::Float32 = 1e-3
    verbose::Int = 1
    η0::Float32 = 1e-1
    power_t::Float32 = 0.25
    dim::Int = 2
end

is_classifier(::GridRegressor) = false

function gridparams(m::GridRegressor)
    grid = ["α" => [0.001, 0.01, 0.1, 1, 10]]
    gridparams(grid)
end

function fit!(m::GridRegressor, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack α, λ, max_iter, tol, verbose, η0, power_t, dim = m
    x = reshape(x, size(x, 1), :)
    F, N = size(x, 1), size(x, 2)
    dims = ntuple(i -> i <= dim ? F : NG, 2dim)
    m.W = W = zeros(Float32, dims)
    ŷ, Δ, ∂W = zero(y), zero(y), zero(W)
    fs = CartesianIndices(ntuple(i -> 1:F, dim))
    loss = prevloss = Inf32
    for t in 1:max_iter
        fill!(∂W, 0)
        predict!(ŷ, m, x)
        Δ .= ŷ .- y
        loss = mean(abs2, Δ)
        # loss > prevloss - tol && break
        prevloss = loss
        if verbose == 2 || (verbose == 1 && t % 10 == 0)
            @printf("t: %d, Norm: %.2g, Loss: %.2g\n", t, norm(W), loss)
        end
        Threads.@threads for n in 1:N
            xn = view(x, :, n)
            for f in fs
                g = gridize(xn, f)
                ∂W[f, g] += Δ[n]
            end
        end
        ∂W ./= N
        α > 0 && (∂W .+= α .* W)
        λ > 0 && (∂W .+= λ .* sign.(W))
        W .-= (η0 / t^power_t) .* ∂W
    end
    BSON.bson("model.bson", model = m)
    visualize(m, columns)
    return m
end

function predict!(ŷ, m::GridRegressor, x)
    fill!(ŷ, 0)
    @unpack W, dim = m
    isempty(W) && return ŷ
    F, N = size(x)
    fs = CartesianIndices(ntuple(i -> 1:F, dim))
    Threads.@threads for n in 1:N
        xn = view(x, :, n)
        for f in fs
            g = gridize(xn, f)
            ŷ[n] += W[f, g]
        end
    end
    return ŷ
end

function predict(m::GridRegressor, x::Array{T}) where T
    @unpack W, dim = m
    x = reshape(x, size(x, 1), :)
    ŷ = zeros(T, size(x, 2))
    predict!(ŷ, m, x)
end

modelhash(m::GridRegressor) = hash(m.W)

function visualize(m::GridRegressor, columns)
    @unpack W, dim = m
    columns′ = fill("", size(m.W))
    for I in CartesianIndices(columns′)
        features = [columns[I[i]] for i in 1:dim]
        bins =  [I[i + dim ] for i in 1:dim]
        columns′[I] = join(join.(zip(features, bins), ':'), 'x')
    end
    write_feaimpt(m.W, columns′)
end

gridize(x) = (n½ = NG ÷ 2 - 1; clamp(floor(Int, n½ * x / 1.96) + n½  + 2, 1, NG))

@generated function gridize(x, f::CartesianIndex{N}) where N
    :(CartesianIndex(($([:(gridize(x[f[$i]])) for i in 1:N]...),)))
end