@ECHO OFF

rmdir /S /Q bin\x64
rmdir /S /Q lib\x64
rmdir /S /Q  build

ECHO ===== CMake for 64-bit ======
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vsdevcmd" -arch=x64
mkdir  build 
cd build

cmake -G "Visual Studio 15 Win64" -DCMAKE_CUDA_FLAGS="-arch=sm_35" .. 

msbuild ALL_BUILD.vcxproj /p:Configuration=Debug /p:Platform=x64 
msbuild ALL_BUILD.vcxproj /p:Configuration=Release /p:Platform=x64
cd ..

