add_mlir_dialect_library(MLIRMyDialect
        MyDialect.cpp
        MyTypes.cpp

        DEPENDS
        MyDialectIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRTensorDialect
)
