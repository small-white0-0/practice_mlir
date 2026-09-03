#ifndef MY_CONVERSION_PASS_TD
#define MY_CONVERSION_PASS_TD

include "mlir/Pass/PassBase.td"

def ConversoinMyToBuiltin : Pass<"conversion-my-to-builtin","::mlir::ModuleOp"> {
    let summary = "把my dialect下降到builtin dialect";
    let description = summary;
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::func::FuncDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect",
        "mlir::memref::MemRefDialect",
        "mlir::bufferization::BufferizationDialect",
        "mlir::LLVM::LLVMDialect"
    ];

}

#endif
