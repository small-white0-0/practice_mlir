#include "Conversion/MyToBuiltin.h"

#include <IR/MyOps.h>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace my {
    namespace {
        struct SoftmaxOpConversionPattern : mlir::OpConversionPattern<my::SoftmaxOp> {
            using Base::Base;

            llvm::LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                                                mlir::ConversionPatternRewriter &rewriter) const override {
                const auto loc = op.getLoc();
                const int64_t axis = op.getAxis();

                // 1. 获取转换后的输入
                auto input = adaptor.getOperands()[0];
                auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
                auto outputType = inputType; // 输入输出前后shape和elementType是一样的。

                // 2. 创建空的输出张量 (Destination-Passing Style)
                auto emptyOutput = mlir::tensor::EmptyOp::create(
                    rewriter,
                    loc,
                    outputType.getShape(),
                    outputType.getElementType());

                // 4. 创建 linalg.softmax 算子
                auto softmaxOp = mlir::linalg::SoftmaxOp::create(
                    rewriter,
                    loc,
                    outputType, // 结果类型
                    input, // 输入 (ins)
                    emptyOutput, // 输出初始化 (outs)
                    rewriter.getI64IntegerAttr(axis) // dimension 属性
                );
                // 主动把linalg.softmax算计进行decompose
                // 否则无法使用官方自带的pass进行下降。
                auto decomposeOp = softmaxOp.decomposeOperation(rewriter);
                if (failed(decomposeOp)) {
                    return mlir::failure();
                }
                // 5. 替换原操作
                rewriter.replaceOp(op, *decomposeOp);
                return mlir::success();
            }
        };

        struct DeviceKernelOpConversionPattern : mlir::OpConversionPattern<my::DeviceKernelOp> {
            using Base::Base;

            llvm::LogicalResult matchAndRewrite(DeviceKernelOp op, OpAdaptor adaptor,
                                                mlir::ConversionPatternRewriter &rewriter) const override {
                auto module = op->getParentOfType<mlir::ModuleOp>();
                if (!module) {
                    return mlir::failure();
                }
                // 这里假设symname是可以等价deviceKernelOp的操作的，
                // 也就是symName一样，那么内部操作就完全等价。
                // 实际上根据目前的devicekernelOp的实现，存在问题，实际可能是不完全等价的。
                // 目前暂时就这样了吧。
                const auto symName = op.getSymName();
                auto funOp = module.lookupSymbol(symName);
                if (!funOp) {
                    const auto savedPoint = rewriter.saveInsertionPoint();
                    rewriter.setInsertionPointToEnd(module.getBody());
                    // 创建fun
                    auto inputs = adaptor.getOperands().getTypes();
                    mlir::SmallVector<mlir::Type> outputs;
                    if (getTypeConverter()->convertTypes(op.getResultTypes(), outputs).failed()) {
                        return mlir::failure();
                    }
                    auto funType = mlir::FunctionType::get(getContext(), inputs, outputs);
                    funOp = mlir::func::FuncOp::create(rewriter, op.getLoc(), symName, funType);
                    // 移动block并修改参数
                    if (auto block = rewriter.convertRegionTypes(&op.getRegion(), *getTypeConverter());
                        succeeded(block)) {
                        auto region = &funOp->getRegion(0);
                        rewriter.moveBlockBefore(*block, region, region->end()); //&funOp->getRegion(0).front());
                    } else {
                        llvm::errs() << "convert block argument failed.";
                        return mlir::failure();
                    }
                    // 恢复insertPointer
                    rewriter.restoreInsertionPoint(savedPoint);
                } else if (!mlir::dyn_cast<mlir::func::FuncOp>(funOp)) {
                    return rewriter.notifyMatchFailure(op->getLoc(), llvm::formatv("{} 已经有了，但是不是funcOp", symName));
                }

                // call
                auto callOp = mlir::func::CallOp::create(
                    rewriter,
                    op->getLoc(),
                    mlir::cast<mlir::func::FuncOp>(funOp),
                    adaptor.getOperands());
                rewriter.replaceOp(op, callOp);
                return mlir::success();
            }
        };

        struct ReturnOpConversionPattern : mlir::OpConversionPattern<my::ReturnOp> {
            using Base::Base;

            llvm::LogicalResult matchAndRewrite(my::ReturnOp op, OpAdaptor adaptor,
                                                mlir::ConversionPatternRewriter &rewriter) const override {
                const auto newOp = mlir::func::ReturnOp::create(rewriter, op.getLoc(), adaptor.getOperands());
                rewriter.replaceOp(op, newOp);
                return mlir::success();
            }
        };

        struct BufferCastOpConversionPattern : mlir::OpConversionPattern<my::BufferCast> {
            using Base::Base;

            llvm::LogicalResult matchAndRewrite(BufferCast op, OpAdaptor adaptor,
                                                mlir::ConversionPatternRewriter &rewriter) const override {
                using namespace ::mlir;
                const auto loc = op->getLoc();
                if (op.getNumOperands() == 1) {
                    // 假设 op 是 my::BufferCast，操作数 0 是输入大张量
                    auto input = adaptor.getOperands()[0];
                    auto inputType = cast<RankedTensorType>(input.getType());

                    int64_t currentOffset = 0; // 沿第 0 维的累积偏移
                    SmallVector<Value> results;

                    for (auto resultType: op.getResultTypes()) {
                        // 从结果类型中提取形状（已经是转换后的类型，如 RankedTensorType）
                        auto resTensorType = cast<RankedTensorType>(getTypeConverter()->convertType(resultType));
                        auto shape = resTensorType.getShape();

                        // 假设切分发生在第 0 维
                        int64_t size = shape[0];

                        // 生成 extract_slice
                        auto slice = tensor::ExtractSliceOp::create(
                            rewriter,
                            loc,
                            resTensorType, // 结果类型
                            input, // 源张量
                            /*offsets*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(currentOffset),
                                rewriter.getIndexAttr(0),
                                rewriter.getIndexAttr(0)
                            },
                            /*sizes*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(size),
                                rewriter.getIndexAttr(shape[1]),
                                rewriter.getIndexAttr(shape[2]),
                            },
                            /*strides*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                            }
                        );

                        results.push_back(slice.getResult());
                        currentOffset += size; // 累加偏移
                    }

                    // 用多个 slice 结果替换原 op 的多个结果
                    rewriter.replaceOp(op, results);
                    return mlir::success();
                } else if (op.getNumResults() == 1) {
                    // op 是 my::BufferCast，有多个操作数（小张量）
                    auto operands = adaptor.getOperands();

                    // 1. 获取目标大张量的形状（可以从 op 的唯一结果中提取）
                    auto resultType = cast<
                        RankedTensorType>(getTypeConverter()->convertType(op.getResult(0).getType()));

                    // 2. 创建空的输出张量
                    auto empty = tensor::EmptyOp::create(
                        rewriter,
                        loc,
                        resultType.getShape(),
                        resultType.getElementType());

                    Value currentOutput = empty;
                    int64_t currentOffset = 0;

                    for (auto operand: operands) {
                        auto operandType = cast<RankedTensorType>(operand.getType());
                        auto shape = operandType.getShape();
                        int64_t size = shape[0];

                        // 生成 insert_slice，将当前小张量插入到累积张量的对应位置
                        currentOutput = tensor::InsertSliceOp::create(
                            rewriter,
                            loc,
                            operand, // 源（小张量）
                            currentOutput, // 目标（大张量）
                            /*offsets*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(currentOffset),
                                rewriter.getIndexAttr(0),
                                rewriter.getIndexAttr(0)
                            },

                            /*sizes*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(size),
                                rewriter.getIndexAttr(shape[1]),
                                rewriter.getIndexAttr(shape[2]),
                            },
                            /*strides*/ SmallVector<OpFoldResult>{
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                            }
                        );

                        currentOffset += size;
                    }

                    // 用拼接好的完整张量替换原 op 的唯一结果
                    rewriter.replaceOp(op, currentOutput);
                    return mlir::success();
                } else {
                    return mlir::failure();
                }
            }
        };

        struct ConstantOpConversionPattern : mlir::OpConversionPattern<my::ConstantOp> {
            using Base::Base;

            llvm::LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                                                mlir::ConversionPatternRewriter &rewriter) const override {
                auto newOp = mlir::arith::ConstantOp::create(rewriter, op.getLoc(), op.getValue());
                rewriter.replaceOp(op, newOp);
                return mlir::success();
            }
        };
    }

    void initMyToBuiltinTypeConvert(mlir::TypeConverter &typeConverter) {
        typeConverter.addConversion([](MyTensorType t) {
            return mlir::RankedTensorType::get(t.getShape(), t.getElementType());
        });
        // 后面两个materialize方法，其实添加的unrealizedConversionCastOp. mlir在没有注册的情况下会默认使用该Op。
        // 写出来只是为了展示自定义materialization使用方法。
        typeConverter.addSourceMaterialization(
            [&](mlir::OpBuilder &builder, mlir::Type resultType, mlir::ValueRange inputs,
                mlir::Location loc) -> mlir::Value {
                if (inputs.size() != 1) return nullptr;
                return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
            });
        typeConverter.addTargetMaterialization(
            [&](mlir::OpBuilder &builder,
                mlir::Type resultType,
                mlir::ValueRange inputs,
                mlir::Location loc) -> mlir::Value {
                if (inputs.size() != 1)
                    return nullptr;
                return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
            });
    }

    void populateMyToBuiltinPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns) {
        patterns.add<
            SoftmaxOpConversionPattern,
            DeviceKernelOpConversionPattern,
            BufferCastOpConversionPattern,
            ConstantOpConversionPattern,
            ReturnOpConversionPattern
        >(
            typeConverter, patterns.getContext());
    }
}
