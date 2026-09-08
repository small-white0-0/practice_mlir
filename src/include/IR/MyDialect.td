#ifndef MY_DIALECT_TD
#define MY_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/PatternBase.td"

def My_Dialect : Dialect {
  let name = "my";
  let cppNamespace = "my";
  let extraClassDeclaration = [{
    void registerTypes();
    void registerOps();
    void registerAttrs();
  }];
  // 启用默认生成的 TypePrinterParser，不然会无法打印Types
  let useDefaultTypePrinterParser = true;
  // 启用默认生成的 AttrPrinterParser，不然会无法打印Attr
  let useDefaultAttributePrinterParser = true;
}

// ===============================================
// Type
class My_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<My_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName = My_Dialect.name # "." # typeMnemonic;
}

def My_TensorType : My_Type<"MyTensor", "my_tensor", [ShapedTypeInterface],"::mlir::TensorType"> {
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
def AnyMyTensor : Type<CPred<"::llvm::isa<::my::MyTensorType>($_self)">>;

// ===============================================
// Interface
class My_AttrInterface<string name,list<Interface> baseInterfaces = []>
    :AttrInterface<name,baseInterfaces>{
  let cppNamespace = "::my";
}

def DistributeParallelAttrInterface : My_AttrInterface<"DistributeParallelAttrInterface",[]> {
  let description = "标记数据并行的属性";
}

def DataParallelAttrInterface: My_AttrInterface<"DataParallelAttrInterface",[DistributeParallelAttrInterface]> {
  let description = "数据并行属性";
  let methods = [
    InterfaceMethod<"并行数量","int64_t","getDP">,
    InterfaceMethod<"设备id","::mlir::ArrayRef<int64_t>","getDeviceIds">
  ];
}

class My_OpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name,baseInterfaces> {
    let cppNamespace = "::my";
}

def DistributeParallelOpInterface : My_OpInterface<"DistributeParallelOpInterface"> {
  let description = "可并行化的Op";
  let methods = [
    InterfaceMethod<"执行并行化转换",
        "::mlir::LogicalResult",
        "applyDistribution",
        (ins "::my::DistributeParallelAttrInterface":$disAttr)>,
    InterfaceMethod<"判断是否可以支持该并行化",
        "bool",
        "supportsDistribution",
        (ins "::my::DistributeParallelAttrInterface":$disAttr)>
  ];
}

def FusionRegionOpInterface : My_OpInterface<"FusionRegionOpInterface"> {
    let methods = [
        StaticInterfaceMethod<"合并Ops","mlir::LogicalResult","FusionOps",
            (ins "::mlir::RewriterBase&":$rewriter, "mlir::ArrayRef<mlir::Operation*>":$ops,"::mlir::Location":$loc)>
    ];
}

// ===============================================
// Attribute
class My_Attribute<string name,  string attrMnemonic, list<Trait> traits = []>
    : AttrDef<My_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def My_DataParallelism : My_Attribute<"DataParallelism","DP",[DeclareAttrInterfaceMethods<DataParallelAttrInterface>]> {
  let parameters = (ins
    "int64_t":$dpNum,
    ArrayRefParameter<"int64_t">:$deviceIds
  );
  let assemblyFormat = [{
  `<` $dpNum `:` $deviceIds `>`
  }];
}

// ===============================================
// Op
class My_Op<string mnemonic, list<Trait> traits = []>
    : Op<My_Dialect, mnemonic, traits> {
  let summary = cppNamespace#opName#"op";

  let description = "$_name op";
}

class My_UnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : My_Op<mnemonic, traits> {
  let arguments = !con((ins OperandType:$input),attributes);
  let results = (outs resultType:$output);
}

class My_BinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : My_Op<mnemonic, traits> {
  let arguments = !con((ins OperandType:$lhs, OperandType:$rhs),attributes);
  let results = (outs resultType:$output);
}

def My_ConstantOp : My_Op<"constant", []> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyMyTensor:$output);
}


def My_SoftmaxOp : My_UnaryOp<"softmax",AnyMyTensor, AnyMyTensor, [DeclareOpInterfaceMethods<DistributeParallelOpInterface>], (ins I64Attr:$axis)>{
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis) ,
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr($_builder.getIntegerType(64), axis);
                $_state.addTypes(input.getType());
            }]>

    ];
}


def My_ExpOp : My_UnaryOp<"exp",AnyMyTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input) ,
            [{
                $_state.addOperands(input);
                $_state.addTypes(input.getType());
            }]>
    ];
}


def My_AddOp : My_BinaryOp<"add",AnyMyTensor>;
def My_SubOp : My_BinaryOp<"sub",AnyMyTensor>;
def My_MulOp : My_BinaryOp<"mul",AnyMyTensor>;
def My_DivOp : My_BinaryOp<"div",AnyMyTensor>;


def My_PrintOp: My_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}

def My_DeviceKernelOp: My_Op<"device_region",
    [DeclareOpInterfaceMethods<FusionRegionOpInterface,["FusionOps"]>,AffineScope, AutomaticAllocationScope,IsolatedFromAbove,
        OpAsmOpInterface ]# GraphRegionNoTerminator.traits>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64Attr:$device_id,
        Variadic<AnyType>:$args
    );

    let regions = (region SizedRegion<1>:$region);

    let results = (outs Variadic<AnyType>:$result);

}

def My_BufferCast : My_Op<"buffer_cast",[]> {
    let summary = "切分和合并";
    let arguments = (ins
        Variadic<AnyMyTensor>: $operands,
        DistributeParallelAttrInterface: $disAttr
    );
    let results = (outs Variadic<AnyMyTensor>: $results);
    // 生成规范化的方法，用于去除不必要的cast
    let hasCanonicalizeMethod = true;
}

def My_ReturnOp : My_Op<"return_op"> {
    let summary = "return operation";
    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif
