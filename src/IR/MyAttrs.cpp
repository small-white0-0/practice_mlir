#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "IR/MyAttrs.h"
#define GET_ATTRDEF_CLASSES
#include "IR/MyAttrs.cpp.inc"


namespace my {
    void MyDialect::registerAttrs() {
        addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/MyAttrs.cpp.inc"
        >();
    }

    int64_t DataParallelismAttr::getDP() const {
        return this->getDpNum();
    }
}
