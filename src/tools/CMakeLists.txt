set(LLVM_OPTIONAL_SOURCES
        null.cpp
)

set(LLVM_LINK_COMPONENTS
        Core
        Support
        AsmParser
)

set(LIBS

        MLIROptLib
        MLIRRegisterAllDialects
        MLIRRegisterAllExtensions
        MLIRRegisterAllPasses

        # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
        # registered directly in mlir-opt.cpp.
        MLIRToLLVMIRTranslationRegistration
)

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
add_mlir_library(MyOptMain
        my-opt.cpp

        EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MyOptMain PUBLIC
        ${LIBS}
)

add_executable(my-opt)
target_sources(my-opt PRIVATE my-opt.cpp)
target_link_libraries(my-opt PRIVATE
        ${LIBS}
        ${MY_LIBS}
)
