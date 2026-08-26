#include "IR/MyTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
/// parse实现需要的类型导入
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "IR/MyDialect.h"

#define GET_TYPEDEF_CLASSES
#include "IR/MyTypes.cpp.inc"

namespace my {
    void MyDialect::registerTypes() {
        addTypes<
#define GET_TYPEDEF_LIST
#include "IR/MyTypes.cpp.inc"
        >();
    }

    ::mlir::Type MyTensorType::parse(::mlir::AsmParser &parser) {
        if (parser.parseLess()) return {};

        llvm::SmallVector<int64_t, 4> dimensions;
        if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true,
                                      /*withTrailingX=*/true))
            return {};
        Type elementType;
        if (parser.parseType(elementType)) return {};
        if (parser.parseComma()) return {};
        int64_t deviceId = 0;
        if (parser.parseInteger(deviceId)) return {};
        if (parser.parseGreater()) return {};

        // Check that array is formed from allowed types.
        return parser.getChecked<MyTensorType>(parser.getContext(), dimensions,
                                               elementType, deviceId);
    }

    void MyTensorType::print(mlir::AsmPrinter &printer) const {
        printer << "<";
        for (int64_t dim: getShape()) {
            if (dim < 0) {
                printer << "?" << 'x';
            } else {
                printer << dim << 'x';
            }
        }
        printer.printType(getElementType());
        printer << "," << getDeviceId() << ">";
    }
}
