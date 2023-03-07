# dcgan_sprite_ai
Sprite generation tool using a DCGAN

## Pre-Requisite
You will need msys2 for this to work. I plan to make powershell scripts that do the same thing but I rely on a psuedo linux commands on windows. This should work with windows 10+

## How to compile
This can be done using cmake itself or by running the auto_compile.sh script in msys2 terminal

### Using auto_compile.sh
This script assumes visual studio 2022 is installed onto your computer but it will do the initial build and copy the neccessary files to where the executable will be.

### Using cmake
If wanting to use cmake directly which could be useful if you do not have visual studio 2022 this is an option.

You will need to create a new directory called **build** (mkdir build). You will then need to cd into the build folder and call **cmake -G "Visual Studio <version> <year>" ..** . An
example of this would look like this...

**cmake -G "Visual Studio Visual Studio 17 2022" ..**

The cmake call will download the neccessary pytorch files and extract them into *<project_root>/libs*

You will then need to build the application and you can do so in 2 ways.
1. Open the visual studio solution file (spriteai.sln) which will open Visual Studio IDE and you can build and code from there
2. call this on the terminal **cmake --build . --config debug/release** (you choose debug or release)
3. should also be able to open the project using CLion as well by opening project where the CMakeLists.txt file is though you need to make sure CLion is set to use the visual studio compiler

# Python Setup
Soon to be written!