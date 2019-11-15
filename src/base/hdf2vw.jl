#!/usr/env/bin julia
using Distributed
@everywhere using Printf, Random, Parameters
@everywhere using ArgParse, HDF5, ProgressMeter

setting = ArgParseSettings()
@add_arg_table setting begin
    "--shuffle"
        arg_type = Bool
        default = false
    "--dst"
        arg_type = String
        default = "data.vw"
    "h5"
        required = true
end
@unpack h5, dst, shuffle = parse_args(ARGS, setting)

@everywhere chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

@everywhere function part(x, n = myid() - 1, N = nworkers(); dim = ndims(x))
    (n < 1 || size(x)[dim] < N) && return x
    is = chunk(1:size(x, dim), N)
    i = UnitRange(extrema(is[n])...)
    inds = ntuple(x -> x == dim ? i : (:), ndims(x))
    view(x, inds...)
end

@everywhere function tovw(x, y, w, shuffle)
    x = reshape(x, size(x, 1), :)
    dst = @sprintf("/dev/shm/%s.vw", randstring())
    fid = open(dst, "w")
    js = shuffle ? randperm(length(y)) : 1:length(y)
    for j in js
        print(fid, y[j], ' ')
        print(fid, w[j], " |f ")
        @inbounds for i in 1:size(x, 1)
            @printf(fid, "%d:%.4g ", i - 1, x[i, j])
        end
        println(fid)
    end
    close(fid)
    return dst
end

srcs = @showprogress "tovw..." pmap(1:100) do n
    h5open(h5) do fid
        x = part(readmmap(fid["x"]), n, 100)
        y = part(readmmap(fid["y"]), n, 100)
        w = part(readmmap(fid["w"]), n, 100)
        tovw(x, y, w, shuffle)
    end
end
shuffle && shuffle!(srcs)

open(dst, "w") do fid
    @showprogress "pd.concat..." for src in srcs
        write(fid, open(read, src))
        rm(src, force = true)
    end
end

print('\n', dst)
