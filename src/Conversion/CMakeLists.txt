add_mlir_conversion_library(MLIRMyConversion
        MyConversionPass.cpp
        MyToBuiltin.cpp

        DEPENDS
        MyConversionPassIncGen


        LINK_LIBS PUBLIC
        MLIRMyDialect
        MLIRMyPasses
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        # 内置的dialect
        # 核心计算
        MLIRArithDialect        # add/sub/mul/div/constant/cmp
        MLIRMathDialect         # exp, log, pow 等（softmax 需要）

        # 控制流与函数
        MLIRFuncDialect         # func.func, func.call, func.return
        MLIRSCFDialect          # for/if（DeviceKernelOp 的 region 需要展开）
        MLIRControlFlowDialect  # scf 最终降到 cf，再降到 llvm

        # 内存与数据类型
        MLIRTensorDialect       # MyTensorType → builtin.tensor 的中间态
        MLIRMemRefDialect       # 最终内存表示（buffer/tensor 的落地）
        MLIRBufferizationDialect # one-shot bufferization（tensor → memref）

        # 最终目标（如果需要出 LLVM IR）
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation  # mlir-translate 用
)