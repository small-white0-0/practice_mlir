#include <iostream>

#include "IR/MyDialect.h"
#include "IR/MyTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    const mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    // 加载 MyDialect
    if (const auto dialect = context.getOrLoadDialect<my::MyDialect>(); !dialect) {
        std::cerr << "Failed to load MyDialect" << std::endl;
        return 1;
    }
    // 构建一个 tensorType 类型
    my::MyTensorType tensorType = my::MyTensorType::get(&context, {2, 3}, mlir::Float32Type::get(&context));
    tensorType.dump(); // 输出类型信息
    // 动态的tensorType
    auto dy_tensorType = my::MyTensorType::get(&context, {mlir::ShapedType::kDynamic, 3},
                                               mlir::Float32Type::get(&context));
    dy_tensorType.dump(); // 输出动态类型信息

    std::cout << "Hello, World!" << std::endl;

    return 0;
}
