#ifndef MY_PASSES_TD
#define MY_PASSED_TD

include "mlir/Pass/PassBase.td"

def MarkDistributeParallelParametersPass : Pass<"mark-distribute-parallel-parameters","::mlir::ModuleOp"> {
  let summary = "标记全局并行参数";
  let description = [{
    "标记全局并行参数。";
  }];
  let dependentDialects = [
    "::my::MyDialect",
    "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"DPNums", "DP", "std::int64_t", /*default=*/"1", "DPNums des">,
    Option<"TPNums", "TP", "std::int64_t", /*default=*/"1", "TPNums des">
  ];
  let statistics = [
    Statistic<"EPNums", "ep-nums", "Number of EP">
  ];
}

def ApplyDistributeTransformPass : Pass<"apply-distribute-transform","::mlir::func::FuncOp"> {
    let summary = "根据并行参数进行转换";
    let description = summary;
    let dependentDialects = [
        "my::MyDialect",
        "::mlir::tensor::TensorDialect"
    ];
}

def DeviceRegionFusionPass : Pass<"device-region-fusion","::mlir::func::FuncOp"> {
    let summary = "合并多个op";
    let dependentDialects = [
    "my::MyDialect"
    ];
}

#endif // MY_PASSED_TD
