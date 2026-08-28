add_mlir_dialect_library(MLIRMyDialect
        MyDialect.cpp
        MyTypes.cpp
        MyOps.cpp
        MyAttrs.cpp
        MyAttrInterface.cpp
        MyOpInterfaces.cpp

        DEPENDS
        MyDialectIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRTensorDialect
)
