set(LLVM_TARGET_DEFINITIONS MyConversionPass.td)
# 多加一个name参数，效果可以让生成的register方法变成`registerMyConversionPasses`
mlir_tablegen(MyConversionPass.h.inc -gen-pass-decls -name MyConversion)
add_mlir_generic_tablegen_target(MyConversionPassIncGen)
