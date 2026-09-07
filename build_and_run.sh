#!/bin/bash

set -e
cd "$(dirname "$0")" # 进入脚本所在目录
if test -d ./third_party/install; then
  echo "install 已经存在，跳过mlir构建安装" >&2
else
  echo "install目录不存在，开始构建mlir..."
  rm -rf ./third_party/build
  cmake -B ./third_party/build -S ./third_party
  cmake --build ./third_party/build --target install -j$(nproc --ignore 2) # 留出2个CPU核心给系统使用
fi
echo "mlir构建安装完成，开始构建项目..."

cmake -B build -S .
cmake --build build -j$(nproc --ignore 2) # 留出2个CPU核心给系统使用

if [ -z "${CODE_MLIR}"]; then
  CODE_MLIR="./code1.mlir"
fi
echo "CODE_MLIR uses ${CODE_MLIR}"
echo "开始执行...."

echo "======= Test 1: 展示My Ops ======"
./build/src/practice_mlir 1
echo "======= Test 2: 展示使用Pass进行算子融合 ======"
./build/src/practice_mlir 2
echo "======= Test 3: 展示使用ConversionPass进行的方言下降 ======"
./build/src/practice_mlir 3
echo "======= Test 4: 展示全流程的mlir到可执行文件 ======"
./build/src/practice_mlir 4 ${CODE_MLIR} ./build/output.ll
clang++ ./build/output.ll -L./build/src -lruntime -o ./build/a.out
./build/a.out
