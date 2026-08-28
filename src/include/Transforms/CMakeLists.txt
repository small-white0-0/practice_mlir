set(LLVM_TARGET_DEFINITIONS MyPasses.td)
mlir_tablegen(MyPasses.h.inc -gen-pass-decls)
add_mlir_dialect_tablegen_target(MyPassesIncGen)
