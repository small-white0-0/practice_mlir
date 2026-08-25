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
