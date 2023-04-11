#!/bin/bash

# setup input and defaults

if [[ "$MSYSTEM" != "MINGW64" ]]; then
	echo "Need MSYS2 (MINGW64) to run this script. Windows powershell script in the works."
	exit 1
fi

build_type=$1
pytorch_build_type=$2

if [[ -z $build_type ]]; then
	build_type="debug"
fi

echo "build type: $build_type"

if [[ -z $pytorch_build_type ]]; then
	pytorch_build_type="release"
fi

echo "build type: $pytorch_build_type"

# create build folder

mkdir libs
mkdir papp

mkdir build
cd build

# generate amd build

if [ "$pytorch_build_type" == "debug" ]; then
	cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug ..
else
	cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release ..
fi

if [ $? != 0 ]; then
    cmake_version=$(cmake --version)
    if [ -z $cmake_version ]; then
		pacman -S mingw-w64-x86_64-cmake
	else
		echo "cmake command error!"
	fi
fi

cmake --build . --config $build_type
