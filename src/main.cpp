#include <iostream>

#include "IR/MyDialect.h"
#include "IR/MyTypes.h"
#include "IR/MyOps.h"
#include "IR/MyAttrs.h"
#include "key.h"
#include "Transforms/MyPasses.h"
#include "Conversion/MyConversionPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

int test0() {
    return 0;
}

int test1() {
    const mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    // 加载 MyDialect
    if (const auto dialect = context.getOrLoadDialect<my::MyDialect>(); !dialect) {
        std::cerr << "Failed to load MyDialect" << std::endl;
        return 1;
    }
    {
        // 构建一个 tensorType 类型
        my::MyTensorType tensorType = my::MyTensorType::get(&context, {2, 3}, mlir::Float32Type::get(&context));
        tensorType.dump(); // 输出类型信息
        // 动态的tensorType
        auto dy_tensorType = my::MyTensorType::get(&context, {mlir::ShapedType::kDynamic, 3},
                                                   mlir::Float32Type::get(&context));
        dy_tensorType.dump(); // 输出动态类型信息
    }
    // ==================================
    // 测试ops
    // ==================================
    {
        auto builder = mlir::OpBuilder(&context);
        auto loc = builder.getUnknownLoc();

        // 创建一个ModuleOp
        auto moduleOp = mlir::ModuleOp::create(builder, loc);
        // 设置builder的插入点到moduleOp的body
        builder.setInsertionPointToStart(moduleOp.getBody());

        // ConstantOp
        const auto f32 = mlir::Float32Type::get(&context);
        const auto value = mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get({2, 2}, f32),
            {1.0f, 1.0f, 1.0f, 1.0f}
        );
        const auto value_tensorType = my::MyTensorType::get(&context, {2, 2}, f32);
        const auto value_tensorType1 = my::MyTensorType::get(&context, {2, 2}, f32, 1);
        value.dump();
        auto const_v1 = my::ConstantOp::create(builder, loc, value_tensorType, value);
        auto const_v2 = my::ConstantOp::create(builder, loc, value_tensorType, value);
        auto const_v3 = my::ConstantOp::create(builder, loc, value_tensorType1, value);
        auto const_v4 = my::ConstantOp::create(builder, loc, value_tensorType1, value);
        llvm::errs() << "Const tensor in divece 0 :\n";
        const_v1->dump();
        llvm::errs() << "Const tensor in divece 1 :\n";
        const_v3->dump();
        // Buffer Op
        auto buffer_op = my::BufferOp::create(builder, loc, mlir::ValueRange{const_v1, const_v3});
        llvm::errs() << "Buffer Op :\n";
        buffer_op->dump();
        // GetTensor Op
        auto get_tensor_op_1 = my::GetTensorOp::create(builder, loc, value_tensorType, buffer_op, 0);
        auto get_tensor_op_2 = my::GetTensorOp::create(builder, loc, value_tensorType1, buffer_op, 1);
        llvm::errs() << "Get Tensor Op :\n";
        get_tensor_op_1->dump();
        get_tensor_op_2->dump();
        // Softmax op
        auto softmax_op = my::SoftmaxOp::create(builder, loc, get_tensor_op_1, 1);
        llvm::outs() << "Softmax Op :\n";
        softmax_op->dump();
        auto exp_op = my::ExpOp::create(builder, loc, get_tensor_op_2);
        llvm::errs() << "Exp Op :\n";
        exp_op->dump();
        // all to all op
        auto out_buffer_op = my::BufferOp::create(builder, loc, mlir::ValueRange{const_v2, const_v4});
        auto all_to_all_op = my::AllToAllOp::create(builder, loc, buffer_op, out_buffer_op);
        llvm::errs() << "All to All Op :\n";
        all_to_all_op->dump();
        moduleOp.dump();
    }
    // 测试属性
    {
        auto builder = mlir::OpBuilder(&context);
        const auto loc = builder.getUnknownLoc();
        auto moduleOp = mlir::ModuleOp::create(builder, loc);
        // builder.setInsertionPointToStart(moduleOp->getBody());  // 会报错。
        builder.setInsertionPointToStart(moduleOp.getBody());
        const auto f32 = mlir::Float32Type::get(&context);
        const auto tensorType = my::MyTensorType::get(&context, {2, 3}, f32);
        auto const_v = my::ConstantOp::create(
            builder,
            loc,
            tensorType,
            mlir::DenseElementsAttr::get(
                mlir::RankedTensorType::get({2, 2}, f32),
                {1.0f, 1.0f, 1.0f, 1.0f})
        );
        auto softmax_op = my::SoftmaxOp::create(builder, loc, const_v, 1);
        llvm::errs() << "bare Softmax Op :\n";
        softmax_op->dump();
        const auto dpAttr = my::DataParallelismAttr::get(
            &context,
            3,
            {0, 1, 2}
        );
        softmax_op->setAttr(my::KDPAttrName, dpAttr);
        llvm::errs() << "added attribute  Softmax Op :\n";
        softmax_op->dump();
        llvm::errs() << "try read dpattr （as my::DataParallelAttrInterface)  of softmax Op\n";
        const auto readDpAttr = softmax_op->getAttr(my::KDPAttrName);
        if (const auto readDpAttr1 = mlir::cast_or_null<my::DataParallelAttrInterface>(readDpAttr)) {
            const auto dp = readDpAttr1.getDP();
            const auto devices = readDpAttr1.getDeviceIds();
            llvm::errs() << "my::DataParallelAttrInterface::getDP() = " << dp << "\n";
            llvm::errs() << "my::DataParallelAttrInterface::getDeviceIds() = ";
            for (const auto device: devices) {
                llvm::errs() << device << ",";
            }
            llvm::errs() << "\n";
        }
    }
    std::cout << "Hello, World!" << std::endl;
    return 0;
}


mlir::ModuleOp getModule(mlir::OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = mlir::ModuleOp::create(builder, loc, "My");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
            my::MyTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
            mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
    auto func =
            mlir::func::FuncOp::create(builder, loc, my::KEntryPointName, func_type);

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // Softmax Op
    mlir::Value softmax_op = my::SoftmaxOp::create(builder,
                                                   loc, block->getArgument(0), 1);
    softmax_op = my::SoftmaxOp::create(builder, loc, softmax_op, 1);
    mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{softmax_op});
    return module;
}

mlir::ModuleOp getModule1(mlir::OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = mlir::ModuleOp::create(builder, loc, "My");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
            my::MyTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
            mlir::FunctionType::get(context, {}, {});
    auto func =
            mlir::func::FuncOp::create(builder, loc, my::KEntryPointName, func_type);

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // const_v
    std::vector<float> values(
        dy_dim * dy_dim * 24,
        1.0f);
    auto const_v = my::ConstantOp::create(
        builder,
        loc,
        dy_tensor_type,
        mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(dy_tensor_type.getShape(), f32),
            mlir::ArrayRef<float>(values)
        ));
    // Softmax Op
    mlir::Value softmax_op = my::SoftmaxOp::create(builder,
                                                   loc, const_v, 1);
    // softmax_op = my::SoftmaxOp::create(builder, loc, softmax_op, 1);
    mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{});
    return module;
}

int test2() {
    const mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    if (!context.getOrLoadDialect<my::MyDialect>()) {
        llvm::outs() << "my::MyDialect not loaded\n";
        return 1;
    }
    if (!context.getOrLoadDialect<mlir::func::FuncDialect>()) {
        llvm::outs() << "mlir::func::FuncDialect not loaded\n";
        return 1;
    }

    mlir::OpBuilder builder(&context);
    auto module = getModule(builder);
    llvm::errs() << "bare module:\n";
    module.dump();
    mlir::PassManager pm(&context);
    pm.addPass(my::createMarkDistributeParallelParametersPass({.DPNums = 3, .TPNums = 1}));
    pm.addNestedPass<mlir::func::FuncOp>(my::createApplyDistributeTransformPass());
    pm.addPass(mlir::createCanonicalizerPass()); // 调用op定义的规范化方法，这个一定要在ApplyDistributeTransformPass后面注册
    pm.addNestedPass<mlir::func::FuncOp>(my::createDeviceRegionFusionPass()); // 对并行化后的op收集到fusionOp中
    if (pm.run(module).failed()) {
        llvm::errs() << "pass failed\n";
        return 1;
    }
    llvm::errs() << "pass succeeded\n";
    llvm::errs() << "transformed module:\n";
    module.dump();
    return 0;
}

int test3() {
    const mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    // 注册内置的所有dialect
    mlir::registerAllDialects(context);
    if (!context.getOrLoadDialect<my::MyDialect>()) {
        llvm::outs() << "my::MyDialect not loaded\n";
        return 1;
    }
    if (!context.getOrLoadDialect<mlir::func::FuncDialect>()) {
        llvm::outs() << "mlir::func::FuncDialect not loaded\n";
        return 1;
    }

    mlir::OpBuilder builder(&context);
    auto module = getModule1(builder);
    llvm::errs() << "bare module:\n";
    module.dump();
    mlir::PassManager pm(&context);
    pm.addPass(my::createMarkDistributeParallelParametersPass({.DPNums = 3, .TPNums = 1}));
    pm.addNestedPass<mlir::func::FuncOp>(my::createApplyDistributeTransformPass());
    pm.addPass(mlir::createCanonicalizerPass()); // 调用op定义的规范化方法，这个一定要在ApplyDistributeTransformPass后面注册
    pm.addNestedPass<mlir::func::FuncOp>(my::createDeviceRegionFusionPass()); // 对并行化后的op收集到fusionOp中
    pm.addPass(my::conversion::createConvertMyToBuiltin()); // 执行ir conversion
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass()); // 对unrealizedcastop的规范化的消除，但是，convert插入该操作会延迟，对于一些不需要的情况会自动删除。


    if (pm.run(module).failed()) {
        llvm::errs() << "pass failed\n";
        return 1;
    }
    llvm::errs() << "final module:\n";
    module.dump();

    return 0;
}


int main() {
    // 根据T调用test1、test2.

#define T 2

#define PASTE2(a, b) a ## b
#define PASTE(a, b) PASTE2(a, b)
#define Call() PASTE(test, T)()
    return Call();
}
