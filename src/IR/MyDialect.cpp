#include "IR/MyDialect.h"
#include "IR/MyDialect.cpp.inc"

namespace my {
    void MyDialect::initialize() {
        llvm::outs() << "MyDialect::initialize\n";
        registerTypes();
        registerOps();
    }
}
