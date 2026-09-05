#ifndef PRACTICE_MLIR_MYCONVERSIONPASS_H
#define PRACTICE_MLIR_MYCONVERSIONPASS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

namespace my::conversion {
    void MyToLLVMPipelineBuilder(mlir::OpPassManager &pm);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/MyConversionPass.h.inc"
}


#endif //PRACTICE_MLIR_MYCONVERSIONPASS_H
