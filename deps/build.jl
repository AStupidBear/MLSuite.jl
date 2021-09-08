using Pkg, BinDeps, PyCall
using PyCall: python, conda, Conda
using BinDeps: generate_steps, getallproviders, lower, PackageManager

!Sys.islinux() && exit()

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

common = Dict("wget" => wget, "git" => git, "make" => make)
provides(AptGet, Dict(common..., "g++" => gcc, "default-jre" => jre, "cmake" => cmake))
provides(Yum, Dict(common..., "gcc-c++" => gcc,  "java-1.8.0-openjdk-headless" => jre, "cmake3" => cmake))

for dep in bindeps_context.deps
    dp, opts = getallproviders(dep, PackageManager)[1]
    cmd = lower(generate_steps(dep, dp, opts)).steps[1]
    i = findfirst(x -> x == "install", cmd.exec)
    insert!(cmd.exec, i + 1, "-y")
    println(cmd)
    try run(cmd) catch end
end

for pkg in ["scikit-learn", "pandas", "h2o", "lightgbm",  "xgboost", "unidecode", "pyarrow", 
            "thundersvm", "tensorflow-ranking", "tensorflow", "treelite", "treelite_runtime"]
    # "catboost==0.16.5"
    try
        run(`$python -m pip install $pkg`)
    catch e
        println(e)
    end
end

buildsh = joinpath(@__DIR__, "build.sh")
ENV["PYTHON"] = python
ENV["JULIA_DEPOT_PATH"] = DEPOT_PATH[1]
run(`bash $buildsh`)
