#ifndef MY_DIALECT_TD
#define MY_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

def My_Dialect : Dialect {
  let name = "my";
  let cppNamespace = "my";
  let extraClassDeclaration = [{
    void registerTypes();
  }];
  // 启用默认生成的 TypePrinterParser，不然会无法打印Types
  let useDefaultTypePrinterParser = true;
}

class My_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<My_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = My_Dialect.name # "." # typeMnemonic;
}

def My_TensorType : My_Type<"MyTensor", "my_tensor", []> {
  let description = [{
    A tensor type with a shape and an element type.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    "int64_t":$deviceId
  );
  let hasCustomAssemblyFormat = true;
//  let assemblyFormat = [{
//    `<` $shape `:` $elementType `>`
//  }];
  let builders =[
    TypeBuilder<(ins "::mlir::ArrayRef<int64_t>":$shape,"::mlir::Type":$elementType),
    "return $_get($_ctxt, shape, elementType, 0);">
  ];
}

#endif
