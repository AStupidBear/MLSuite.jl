using Pkg, BinDeps, PyCall
using PyCall: python, conda, Conda
using BinDeps: generate_steps, getallproviders, lower, PackageManager

!Sys.islinux() && exit()

if conda && Conda.version("python") >= v"3.7"
    # treelite does not support python3.7
    Conda.add("python=3.6")
    Pkg.build("PyCall")
end

run(`$python -m pip install scikit-learn pandas h2o lightgbm xgboost unidecode pyarrow thundersvm treelite`)
run(`$python -m pip install catboost==0.16.5 tensorflow-ranking==0.1.2 tensorflow==1.12`)

if isnothing(Sys.which("sudo")) # in docker
    try run(`apt update`) catch end
    try run(`yum update`) catch end
end

@BinDeps.setup

wget = library_dependency("wget")
git = library_dependency("git")
make = library_dependency("make")
cmake = library_dependency("cmake")
gcc = library_dependency("gcc")
jre = library_dependency("jre")

common = Dict("wget" => wget, "git" => git, "make" => make, "cmake" => cmake)
provides(AptGet, Dict(common..., "g++" => gcc, "default-jre" => jre))
provides(Yum, Dict(common..., "gcc-c++" => gcc,  "java-1.8.0-openjdk" => jre))

for dep in bindeps_context.deps
    dp, opts = getallproviders(dep, PackageManager)[1]
    cmd = lower(generate_steps(dep, dp, opts)).steps[1]
    i = findfirst(x -> x == "install", cmd.exec)
    insert!(cmd.exec, i + 1, "-y")
    run(cmd)
end

buildsh = joinpath(@__DIR__, "build.sh")
ENV["PYTHON"] = python
ENV["JULIA_DEPOT_PATH"] = DEPOT_PATH[1]
run(`bash $buildsh`)