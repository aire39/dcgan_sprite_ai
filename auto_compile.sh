#!/bin/bash

# setup input and defaults

build_type=$1

if [[ !build_type ]]; then
	build_type="debug"
fi

# create build folder

mkdir libs
mkdir papp

mkdir build
cd build

# generate amd build

cmake -G "Visual Studio 17 2022" ..
cmake --build . --config $build_type

# copy needed dll files

if [[ $build_type=="debug" ]]; then
	cp ../libs/pytorch_1_13_0_$build_type/libtorch/lib/*.dll Debug/
else
	cp ../libs/pytorch_1_13_0_$build_type/libtorch/lib/*.dll Release/
fi
