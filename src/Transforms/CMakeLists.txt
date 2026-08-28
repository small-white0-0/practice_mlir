add_mlir_dialect_library(MLIRMyPasses
        MyPasses.cpp

        DEPENDS
        MyPassesIncGen

        LINK_LIBS PUBLIC
        MLIRMyDialect
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
)
