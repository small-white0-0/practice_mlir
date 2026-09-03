#ifndef PRACTICE_MLIR_MYTOBUILTIN_H
#define PRACTICE_MLIR_MYTOBUILTIN_H
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir {
    class TypeConverter;
}

namespace my {
    void initMyToBuiltinTypeConvert(mlir::TypeConverter &typeConverter);

    void populateMyToBuiltinPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns);
}

#endif //PRACTICE_MLIR_MYTOBUILTIN_H
