#include <iostream>

#include "IR/MyDialect.h"
#include "mlir/IR/DialectRegistry.h"

int main() {
    const mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    // 加载 MyDialect
    if (const auto dialect = context.getOrLoadDialect<my::MyDialect>(); !dialect) {
        std::cerr << "Failed to load MyDialect" << std::endl;
        return 1;
    }

    std::cout << "Hello, World!" << std::endl;

    return 0;
}
