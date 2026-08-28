//
// Created by tt on 2026/8/29.
//

#ifndef PRACTICE_MLIR_MYPASSES_H
#define PRACTICE_MLIR_MYPASSES_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

namespace my {
// 这个其实能拿到的就只有create方法了。
#define GEN_PASS_DECL
#include "Transforms/MyPasses.h.inc"
}
#endif //PRACTICE_MLIR_MYPASSES_H
