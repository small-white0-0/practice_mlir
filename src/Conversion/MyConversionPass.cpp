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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

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

    void configConversionTarget(mlir::ConversionTarget &target, mlir::TypeConverter &typeConverter) {
        target.addLegalOp<mlir::ModuleOp>();
        target.addLegalDialect<mlir::linalg::LinalgDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::math::MathDialect>();
        // target.addLegalDialect<mlir::func::FuncDialect>(); // 需要执行func的type转换pattern，所以不能直接标记为legal,需要使用动态标记
        target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType());
        });
        target.addDynamicallyLegalOp<mlir::func::CallOp, mlir::func::ReturnOp>([&](mlir::Operation *op) {
            return typeConverter.isLegal(op);
        });
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
        mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
        mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
        mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
        mlir::ConversionTarget target(getContext());
        configConversionTarget(target, typeConverter);
        if (mlir::applyPartialConversion(module, target, mlir::FrozenRewritePatternSet(std::move(patterns))).failed()) {
            module.print(llvm::outs());
            signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
    }

    void MyToLLVMPipelineBuilder(mlir::OpPassManager &pm) {
        pm.addPass(my::conversion::createConvertMyToBuiltin());
        // 下降后优化
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // Tensor → Linalg 转换
        pm.addPass(mlir::createConvertTensorToLinalgPass());
        pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // Bufferization（将张量转换为 MemRef）
        mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        // 对应 --one-shot-bufferize="bufferize-function-boundaries"
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // Linalg → 循环（SCF）
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // 控制流结构化 → 非结构化（CF）
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        // 各种方言 → LLVM 方言
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertIndexToLLVMPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());

        // 收尾：MemRef 完全转换为 LLVM 类型
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

        // 最后的通用优化规范化
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
    }
}
