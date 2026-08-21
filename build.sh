#!/bin/bash

set -e
cd "$(dirname "$0")" # 进入脚本所在目录
if test -d ./install; then
  echo "install 已经存在，确认是否需要重新构建，如果需要请删除build和install目录." >&2
  exit 1
fi
cmake -B build
# 使用install目标是为了把mlir的可执行文件放到./install中。
cmake --build build --target install -j$(nproc --ignore 2) # 留出2个CPU核心给系统使用

