#ifndef MY_DIALECT_TD
#define MY_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

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
def AnyMyTensor : Type<CPred<"::llvm::isa<::my::MyTensorType>($_self)">>;

def My_BufferType : My_Type<"Buffer","buffer",[]>{
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$devices
  );

  let genStorageClass = 1;

  let hasStorageCustomConstructor = 0;

  let assemblyFormat = "`<`$devices`>`";

  let skipDefaultBuilders = 0;

}
def AnyBuffer : Type<CPred<"::llvm::isa<::my::BufferType>($_self)">>;
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


def My_AllToAllOp : My_Op<"all_to_all",[]>{
    let arguments = (ins
        AnyBuffer:$input,
        AnyBuffer:$output
    );
}


def My_BufferOp : My_Op<"buffer",[]>{
    let description = "将 多个device_id 的tensor 组合成 一个 buffer";
    let arguments = (ins
        Variadic<AnyMyTensor>:$tensors
    );
    let results = (outs
        AnyBuffer:$result);
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors) ,
            [{
                $_state.addOperands(tensors);
                ::llvm::SmallVector<int64_t> devices;
                for (auto tensor : tensors) {
                    auto tensor_type =
                        llvm::cast<::my::MyTensorType>(tensor.getType());
                    devices.push_back(tensor_type.getDeviceId());
                }
                $_state.addTypes(::my::BufferType::get($_state.getContext(), devices));
            }]>
    ];
}

def My_GetTensorOp: My_Op<"get_tensor",[]>{
    let description = "从buffer中取出指定device_id的tensor";
    let arguments = (ins
        AnyBuffer:$buffer,
        I64Attr:$device_id
    );
    let results = (outs
        AnyMyTensor:$result);
    let hasVerifier = 1;
}

def My_PrintOp: My_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}
#endif
