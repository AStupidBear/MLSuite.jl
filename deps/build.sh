#!/bin/bash
BIN=$(pwd)/usr/bin
LGBM=$(pwd)/usr/lightgbm
CUDA=$(pwd)/usr/cuda
BOOST=$(pwd)/usr/boost
mkdir -p $BIN

alias proxychains=''

if [ -z "$PYTHON" ]; then
    PYTHON=$(which python3)
fi

if [[ ! -f $BIN/vw ]]; then
    wget -O $BIN/vw http://finance.yendor.com/ML/VW/Binaries/vw-8.20190624
    wget -O $BIN/vw-varinfo https://raw.githubusercontent.com/arielf/weight-loss/master/vw-varinfo2
    chmod +x $BIN/{vw,vw-varinfo}
fi

if [[ ! -f $BIN/ranklib ]]; then
    proxychains wget -O $BIN/ranklib.jar https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.11/RankLib-2.11.jar
    echo '#!/bin/bash' > $BIN/ranklib
    echo 'java -jar $(dirname $0)/ranklib.jar "$@"' >> $BIN/ranklib
    chmod +x $BIN/ranklib
fi

if [[ ! -f $BIN/svm_rank_classify ]]; then
    wget http://download.joachims.org/svm_rank/current/svm_rank_linux64.tar.gz
    tar xvzf svm_rank_linux64.tar.gz -C $BIN && \rm svm_rank_*.tar.gz
fi

if [[ ! -d $CUDA ]]; then
    wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
    bash cuda_*_linux --silent --toolkit --toolkitpath=$CUDA && \rm cuda_*_linux
fi

if [[ ! -d $BOOST ]]; then
    proxychains wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
    tar -zxvf boost_1_66_0.tar.gz && rm boost*.tar.gz
    cd boost_1_66_0 && ./bootstrap.sh --prefix=$BOOST && \
    ./b2 install && cd .. && \rm -rf boost_1_66_0
fi

if [[ ! -f $BIN/lightgbm ]]; then
    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM && mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$BIN \
            -DUSE_MPI=ON -DUSE_GPU=1 \
            -DBOOST_ROOT=$BOOST \
            -DOpenCL_LIBRARY=$CUDA/lib64/libOpenCL.so \
            -DOpenCL_INCLUDE_DIR=$CUDA/include/
    make -j4 && make install && cd ..
    cd python-package && $PYTHON setup.py install --precompile -O2
    cd .. && \rm -rf LightGBM
    ln -s $LGBM/bin/lightgbm $BIN/lightgbm
fi