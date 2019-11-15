export signone

function sortall(xs::AbstractArray...; kw...)
    p = sortperm(first(xs); kw...)
    map(x -> x[p], xs)
end

function sortall!(xs::AbstractArray...; kw...)
    p = sortperm(first(xs); kw...)
    for x in xs
        permute!(x, p)
    end
    return xs
end

function write_feaimpt(w, c; dst = "feaimpt.csv")
    length(w) != length(c) && return
    w, c = sortall(vec(w), vec(c), by = abs, rev = true)
    open(dst, "w") do io
        for i in 1:min(100, length(c))
            write(io, togbk(c[i]), ',')
            println(io, trunc(w[i], digits = 2))
        end
    end
end

function hasgpu()
    try
        run(`nvidia-smi`, wait = false)
        return true
    catch
        return false
    end
end

usegpu() = hasgpu() && get(ENV, "USE_GPU", "1") == "1"

countgpus() = hasgpu() ? count(x -> x == '\n', read(`nvidia-smi -L`, String)) : 0

function to_svm(x, y, w = nothing; group = nothing)
    @imports numpy as np
    if !isnothing(group) && length(group) != length(y)
        qid = np.repeat(1:length(group), group)
    else
        qid = group
    end
    dst = @sprintf("/dev/shm/%s.svmlight", randstring())
    x = reshape(x, size(x, 1), :)
    y, qid = vec(y), vec(qid)
    dump_svmlight_file(x, y, dst, query_id = qid, zero_based = false)
    return dst
end

function to_svm(x)
    x = reshape(x, size(x, 1), :)
    y = zeros(Float32, size(x, 2))
    to_svm(x, y, group = [length(y)])
end

macro NT(xs...)
    xs = [:($x = $x) for x in xs]
    esc(:(($(xs...),)))
end

function bash(str, exe = run)
    exe(`bash -c $str`)
end

parseenv(key, default::String) = get(ENV, string(key), string(default))

function parseenv(key, default::T) where T
    str = get(ENV, string(key), string(default))
    if hasmethod(parse, (Type{T}, String))
        parse(T, str)
    else
        include_string(Main, str)
    end
end

paramgrid(grid::AbstractDict, combine = Iterators.product) =
    [Dict(zip(keys(grid), v)) for v in combine(values(grid)...)]

rmdir(src) = rm(src, recursive = true)

memory(x) = Base.summarysize(x) / 1024^2

function align(x, y)
    iy, Nx, Ny = 1, length(x), length(y)
    for ix in 1:(Nx - Ny + 1)
        for iy in 1:Ny
            if x[iy + ix - 1] != y[iy]
                break
            elseif iy == Ny
                return ix
            end
        end
    end
    return 0
end

macro gc(exs...)
    Expr(:block, [:($ex = 0) for ex in exs]..., :(@eval GC.gc(true))) |> esc
end

macro redirect(src, ex)
    src = src == :devnull ? "/dev/null" : src
    quote
        io = open($(esc(src)), "a")
        o, e = stdout, stderr
        redirect_stdout(io)
        redirect_stderr(io)
        try
            $(esc(ex)); sleep(0.01)
        finally
            flush(io); close(io)
            redirect_stdout(o)
            redirect_stderr(e)
        end
    end
end

Base.vec(::Nothing) = nothing

bra(x) = isnothing(x) ? x : [x]

Base.sign(x::Real, Θ) = ifelse(x < -Θ, oftype(x, -1), ifelse(x > Θ, one(x), zero(x)))

function signone(x::Real, Θ = zero(x))
    y = sign(x, Θ)
    !iszero(y) && return y
    rand([-one(x), one(x)])
end

sigmoid(x) = 1 / (1 + exp(-x))

function softmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims = dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims = dims)
end

export @trys
macro trys(exs...)
    expr = :()
    for ex in exs[end:-1:1]
        expr = :(try $ex; catch e; $expr; end)
    end
    esc(expr)
end