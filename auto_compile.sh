#!/bin/bash

# setup input and defaults

if [[ "$MSYSTEM" != "MINGW64" ]]; then
	echo "Need MSYS2 (MINGW64) to run this script. Windows powershell script in the works."
	exit 1
fi

build_examples=$1

# create build folder

mkdir libs
mkdir papp

mkdir build
cd build

# generate amd build

if [[ -z build_examples ]]; then
  cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=FALSE ..
  cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=FALSE ..
else
  cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=TRUE ..
  cmake -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=TRUE ..
fi

if [ $? != 0 ]; then
    cmake_version=$(cmake --version)
    if [ -z $cmake_version ]; then
		pacman -S mingw-w64-x86_64-cmake
	else
		echo "cmake command error!"
	fi
fi

cmake --build . --config debug
cmake --build . --config release
