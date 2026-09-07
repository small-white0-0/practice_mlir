#include "Conversion/MyToBuiltin.h"

#include "IR/MyOps.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"

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

        struct PrintOpConversionPattern
                : public mlir::OpConversionPattern<my::PrintOp> {
            using OpConversionPattern<my::PrintOp>::OpConversionPattern;

            llvm::LogicalResult matchAndRewrite(
                my::PrintOp op,
                OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const override {
                mlir::Value input = adaptor.getInput();
                mlir::Type inputType = input.getType();

                // --------------------------------------------------------
                // 1. scalar -> memref<1xT>
                //    memref 保持原类型则直接进入下一步
                //    MyTensor(tensor) 构建 materialization
                // --------------------------------------------------------
                mlir::Value printValue = input;
                mlir::MemRefType rankedMemRef;

                if (auto myTensorType = mlir::dyn_cast<my::MyTensorType>(op.getInput().getType())) {
                    auto memrefType = mlir::MemRefType::get(
                        myTensorType.getShape(),
                        myTensorType.getElementType());
                    auto cast = mlir::bufferization::ToBufferOp::create(
                        rewriter,
                        op.getLoc(),
                        memrefType,
                        adaptor.getInput());

                    rankedMemRef = memrefType;
                    printValue = cast.getResult();
                } else if (auto memrefType =
                        mlir::dyn_cast<mlir::MemRefType>(inputType)) {
                    rankedMemRef = memrefType;
                } else {
                    // 这里可以根据你的实际需求限制 scalar 类型
                    if (!mlir::isa<mlir::IntegerType>(inputType) &&
                        !mlir::isa<mlir::FloatType>(inputType)) {
                        return rewriter.notifyMatchFailure(
                            op, "unsupported print type");
                    }

                    rankedMemRef = mlir::MemRefType::get(
                        {}, inputType);

                    auto alloca = mlir::memref::AllocaOp::create(rewriter, op.getLoc(), rankedMemRef);

                    mlir::memref::StoreOp::create(rewriter, op.getLoc(), input, alloca);

                    printValue = alloca;
                }

                // --------------------------------------------------------
                // 2. 根据 element type 选择 runtime
                // --------------------------------------------------------
                mlir::Type elementType =
                        rankedMemRef.getElementType();

                llvm::StringRef runtimeName;

                if (mlir::isa<mlir::Float32Type>(elementType)) {
                    runtimeName = "my_print_f32";
                } else if (mlir::isa<mlir::Float64Type>(elementType)) {
                    runtimeName = "my_print_f64";
                } else if (mlir::isa<mlir::IntegerType>(elementType)) {
                    auto intType =
                            mlir::cast<mlir::IntegerType>(elementType);

                    switch (intType.getWidth()) {
                        case 32:
                            runtimeName = "my_print_i32";
                            break;
                        case 64:
                            runtimeName = "my_print_i64";
                            break;
                        default:
                            return rewriter.notifyMatchFailure(
                                op, "unsupported integer element type");
                    }
                } else {
                    return rewriter.notifyMatchFailure(
                        op, "unsupported element type");
                }

                // --------------------------------------------------------
                // 3. ranked memref -> unranked memref
                // --------------------------------------------------------
                auto unrankedType =
                        mlir::UnrankedMemRefType::get(
                            elementType,
                            rankedMemRef.getMemorySpace());

                mlir::Value unrankedValue = mlir::memref::CastOp::create(
                    rewriter, op.getLoc(), unrankedType, printValue);

                // --------------------------------------------------------
                // 4. runtime 函数类型
                //
                // func @my_print_f32(memref<*xf32>)
                // --------------------------------------------------------
                auto funcType =
                        mlir::FunctionType::get(
                            rewriter.getContext(),
                            {unrankedType},
                            {});

                auto module =
                        op->getParentOfType<mlir::ModuleOp>();

                if (!module)
                    return rewriter.notifyMatchFailure(
                        op, "no enclosing module");

                auto printFunc =
                        module.lookupSymbol<mlir::func::FuncOp>(
                            runtimeName);

                // --------------------------------------------------------
                // 5. 创建 runtime declaration
                // --------------------------------------------------------
                if (!printFunc) {
                    mlir::OpBuilder::InsertionGuard guard(rewriter);

                    rewriter.setInsertionPointToStart(
                        module.getBody());

                    printFunc =
                            rewriter.create<mlir::func::FuncOp>(
                                op.getLoc(),
                                runtimeName,
                                funcType);
                    printFunc.setPrivate();
                    printFunc->setAttr(
                        mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                        mlir::UnitAttr::get(
                            rewriter.getContext()));
                } else if (printFunc.getFunctionType() != funcType) {
                    return rewriter.notifyMatchFailure(
                        op,
                        "existing print runtime has incompatible type");
                }

                // --------------------------------------------------------
                // 6. my.print -> func.call
                // --------------------------------------------------------
                rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
                    op,
                    runtimeName,
                    mlir::TypeRange{},
                    mlir::ValueRange{unrankedValue});
                return mlir::success();
            }
        };

        template<typename SrcOp, typename LinalgOp>
        struct BinaryElementwiseConversionPattern
                : public mlir::OpConversionPattern<SrcOp> {
            using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;

            mlir::LogicalResult matchAndRewrite(
                SrcOp op,
                typename SrcOp::Adaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const override {
                auto lhs = adaptor.getLhs();
                auto rhs = adaptor.getRhs();

                auto lhsType =
                        mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
                auto rhsType =
                        mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());

                if (!lhsType || !rhsType)
                    return rewriter.notifyMatchFailure(
                        op, "operands must be ranked tensors");

                // 当前先要求 shape 完全一致。
                if (lhsType.getShape() != rhsType.getShape())
                    return rewriter.notifyMatchFailure(
                        op, "broadcasting not implemented yet");

                auto resultType =
                        mlir::dyn_cast<mlir::RankedTensorType>(
                            this->getTypeConverter()->convertType(
                                op.getResult().getType()));

                if (!resultType)
                    return rewriter.notifyMatchFailure(
                        op, "result must be ranked tensor");

                auto loc = op.getLoc();

                // 创建 destination。
                auto empty =
                        rewriter.create<mlir::tensor::EmptyOp>(
                            loc,
                            resultType.getShape(),
                            resultType.getElementType());

                // ------------------------------------------------------------
                // 创建具体的 linalg named op
                // ------------------------------------------------------------

                auto linalgOp =
                        LinalgOp::create(
                            rewriter,
                            loc,
                            /*resultTensorTypes=*/mlir::TypeRange{resultType},
                            /*inputs=*/mlir::ValueRange{lhs, rhs},
                            /*outputs=*/mlir::ValueRange{empty});

                // ------------------------------------------------------------
                // 替换原 My Op
                // ------------------------------------------------------------

                rewriter.replaceOp(
                    op,
                    linalgOp.getResultTensors());

                return mlir::success();
            }
        };

        struct ExpOpConversionPattern
                : public mlir::OpConversionPattern<my::ExpOp> {
            using OpConversionPattern<my::ExpOp>::OpConversionPattern;

            mlir::LogicalResult matchAndRewrite(
                my::ExpOp op,
                OpAdaptor adaptor,
                mlir::ConversionPatternRewriter &rewriter) const override {
                auto input = adaptor.getInput();

                auto inputType =
                        mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
                if (!inputType)
                    return rewriter.notifyMatchFailure(
                        op, "expected ranked tensor");

                auto resultType =
                        mlir::dyn_cast<mlir::RankedTensorType>(
                            getTypeConverter()->convertType(op.getResult().getType()));
                if (!resultType)
                    return rewriter.notifyMatchFailure(
                        op, "expected ranked tensor result");

                auto loc = op.getLoc();

                // 创建 destination tensor。
                auto empty = rewriter.create<mlir::tensor::EmptyOp>(
                    loc,
                    resultType.getShape(),
                    resultType.getElementType());

                // linalg.exp(input, output)
                auto exp = rewriter.create<mlir::linalg::ExpOp>(
                    loc,
                    mlir::ValueRange{input},
                    mlir::ValueRange{empty});

                rewriter.replaceOp(
                    op,
                    exp.getResultTensors());

                return mlir::success();
            }
        };
    }

    void initMyToBuiltinTypeConvert(mlir::TypeConverter &typeConverter) {
        typeConverter.addConversion([](mlir::Type t) { return t; });
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
            ReturnOpConversionPattern,
            PrintOpConversionPattern,
            BinaryElementwiseConversionPattern<my::AddOp, mlir::linalg::AddOp>,
            BinaryElementwiseConversionPattern<my::SubOp, mlir::linalg::SubOp>,
            BinaryElementwiseConversionPattern<my::MulOp, mlir::linalg::MulOp>,
            BinaryElementwiseConversionPattern<my::DivOp, mlir::linalg::DivOp>,
            ExpOpConversionPattern
        >(
            typeConverter, patterns.getContext());
    }
}
