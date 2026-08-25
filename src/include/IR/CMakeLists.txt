set(LLVM_TARGET_DEFINITIONS MyDialect.td)
mlir_tablegen(MyDialect.h.inc -gen-dialect-decls -dialect=my)
mlir_tablegen(MyDialect.cpp.inc -gen-dialect-defs -dialect=my)
add_mlir_dialect_tablegen_target(MyDialectIncGen)
