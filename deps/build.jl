using PyCall: python

run(`$python -m pip install scikit-learn scipy pandas h2o lightgbm xgboost
    catboost unidecode pyarrow thundersvm tensorflow_ranking treelite`)
    
buildsh = joinpath(@__DIR__, "build.sh")
ENV["PYTHON"] = python
bash(`bash $buildsh`)