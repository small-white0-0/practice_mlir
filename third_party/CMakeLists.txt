cmake_minimum_required(VERSION 4.2)
project(practice_mlir)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

set(CMAKE_BUILD_TYPE "Release")
# 使用 lld 加快链接速度
set(LLVM_USE_LINKER "lld")
# 启用 RTTI
set(LLVM_ENABLE_RTTI ON)
# 禁用异常处理
set(LLVM_ENABLE_EH OFF)
# 启用断言检查
set(LLVM_ENABLE_ASSERTIONS ON)
# 包含示例代码
set(LLVM_INCLUDE_EXAMPLES ON)
# 包含工具
set(LLVM_INCLUDE_TOOLS ON)
## 需要写入到cache中，才能避免llvm项目中设置的初始值覆盖了。
# 启用 MLIR 项目
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING " ")
# 编译目标后端 Native 是本机架构
set(LLVM_TARGETS_TO_BUILD "Native;AMDGPU" CACHE STRING " ")
add_subdirectory(llvm-project/llvm)
