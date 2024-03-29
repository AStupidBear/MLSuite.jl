#!/bin/bash
MLSUITE=$JULIA_DEPOT_PATH/mlsuite
FAKETIME=$JULIA_DEPOT_PATH/faketime
BIN=$MLSUITE/bin
LGBM=$MLSUITE/lightgbm
CUDA=$MLSUITE/cuda
BOOST=$MLSUITE/boost
mkdir -p $BIN && cd /tmp

if nvidia-smi &> /dev/null; then
    USE_GPU=1
else
    USE_GPU=0
fi

if mpiexec --help &> /dev/null; then
    USE_MPI=1
else
    USE_MPI=0
fi

if [ -z "$PYTHON" ]; then
    PYTHON=$(which python3)
fi

if [ ! -f $BIN/vw ]; then
    wget -O $BIN/vw http://finance.yendor.com/ML/VW/Binaries/vw-8.20190624
    wget -O $BIN/vw-varinfo https://raw.githubusercontent.com/arielf/weight-loss/master/vw-varinfo2
    chmod +x $BIN/{vw,vw-varinfo}
fi

if [ ! -f $BIN/ranklib.jar ]; then
    wget -O $BIN/ranklib.jar https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.11/RankLib-2.11.jar
    echo '#!/bin/bash' > $BIN/ranklib
    echo 'java -jar $(dirname $0)/ranklib.jar "$@"' >> $BIN/ranklib
    chmod +x $BIN/ranklib
fi

if [ ! -f $BIN/svm_rank_classify ]; then
    wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz
    tar xvzf svm_rank_linux64.tar.gz -C $BIN && rm svm_rank_*.tar.gz
fi

if [ ! -d $CUDA ] && [ $USE_GPU == 1 ]; then
    wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
    bash cuda_*_linux --silent --toolkit --toolkitpath=$CUDA && rm cuda_*_linux
fi

if [ ! -d $BOOST ] && [ $USE_GPU == 1 ]; then
    wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
    tar -zxvf boost_1_66_0.tar.gz && rm boost*.tar.gz
    cd boost_1_66_0 && ./bootstrap.sh --prefix=$BOOST && ./b2 install
    cd /tmp && rm -rf boost_1_66_0
fi

if [ ! -f $LGBM/bin/lightgbm ]; then
    git clone --recursive https://github.com/Microsoft/LightGBM
    find LightGBM -name 'rank_metric.hpp' | xargs -n 1 sed -i 's/ \* inverse_max_dcgs_\[i\]\[j\]//g'
    find LightGBM -name 'rank_objective.hpp' | xargs -n 1 sed -i 's/paired_discount \* inverse_max_dcg/paired_discount/g'
    find LightGBM -name 'dcg_calculator.cpp' | xargs -n 1 sed -i 's/1.0 \/ std::log2(2.0 + i)/1.0/g'
    cd LightGBM && mkdir -p build && cd build
    CMAKE=$(which cmake3 &> /dev/null && echo cmake3 || echo cmake)
    $CMAKE .. -DCMAKE_INSTALL_PREFIX=$LGBM \
            -DUSE_MPI=$USE_MPI \
            -DUSE_GPU=$USE_GPU \
            -DBOOST_LIBRARYDIR=$BOOST/lib/ \
            -DBoost_INCLUDE_DIR=$BOOST/include/ \
            -DOpenCL_LIBRARY=$CUDA/lib64/libOpenCL.so \
            -DOpenCL_INCLUDE_DIR=$CUDA/include/
    make -j4 && make install && cd ..
    bash -c "cd $BIN && ln -sfn ../lightgbm/bin/lightgbm lightgbm"
    cd python-package && $PYTHON setup.py install --precompile -O2
    cd /tmp && rm -rf LightGBM
fi

if [ ! -f $FAKETIME/lib/libfaketime.so.1 ]; then
    git clone https://github.com/wolfcw/libfaketime.git
    cd libfaketime/src && PREFIX=$FAKETIME LIBDIRNAME=/lib make install
    cd /tmp && rm -rf libfaketime
fi
