cmake_minimum_required(VERSION 4.2)
project(practice_mlir)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CMAKE_PREFIX_PATH以包含第三方库的安装路径
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/install")
# 添加LLVM和MLIR的CMake Config文件
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
# 添加头文件路径
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# 添加模块路径
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
# 添加模块
include(TableGen)
include(AddLLVM)
include(AddMLIR)

if (NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif ()
add_subdirectory(src)