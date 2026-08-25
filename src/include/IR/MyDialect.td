#ifndef MY_DIALECT_TD
#define MY_DIALECT_TD

include "mlir/IR/DialectBase.td"

def My_Dialect : Dialect {
  let name = "my";
  let cppNamespace = "my";
}



#endif
