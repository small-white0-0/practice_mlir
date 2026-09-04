#include "llvm/Support/Debug.h"
#include "Conversion/MyConversionPass.h"

#include "IR/MyOps.h"
#include "Conversion/MyToBuiltin.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect//Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


namespace my::conversion {
#define GEN_PASS_DEF_CONVERTMYTOBUILTIN
#include "Conversion/MyConversionPass.h.inc"

    struct MyConversionPass : impl::ConvertMyToBuiltinBase<MyConversionPass> {
        // 继承构造函数
        using Base::Base;
    protected:
        // 重载执行
        void runOnOperation() override;
    };

    void configConversionTarget(mlir::ConversionTarget &target) {
        target.addLegalOp<mlir::ModuleOp>();
        target.addLegalDialect<mlir::linalg::LinalgDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::math::MathDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addLegalDialect<mlir::scf::SCFDialect>();
        target.addLegalDialect<mlir::tensor::TensorDialect>();
        target.addLegalDialect<mlir::memref::MemRefDialect>();
        target.addLegalDialect<mlir::bufferization::BufferizationDialect>();
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addIllegalDialect<my::MyDialect>();
    }

    void MyConversionPass::runOnOperation() {
        LLVM_DEBUG(llvm::dbgs() << "Run MyConversionPass on");
        auto module = getOperation();
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv("{0}\n", module.getName()));
        mlir::TypeConverter typeConverter;
        initMyToBuiltinTypeConvert(typeConverter);
        mlir::RewritePatternSet patterns(module.getContext());
        populateMyToBuiltinPatterns(typeConverter, patterns);
        mlir::ConversionTarget target(getContext());
        configConversionTarget(target);
        if (mlir::applyPartialConversion(module, target, mlir::FrozenRewritePatternSet(std::move(patterns))).failed()) {
            module.print(llvm::outs());
            signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
    }
}
