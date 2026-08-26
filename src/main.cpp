#include <iostream>

#include "IR/MyDialect.h"
#include "IR/MyTypes.h"
#include "IR/MyOps.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

int main() {
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
        auto moduleOp = builder.create<mlir::ModuleOp>(loc);
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
        auto const_v1 = builder.create<my::ConstantOp>(loc, value_tensorType, value);
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
    std::cout << "Hello, World!" << std::endl;

    return 0;
}
