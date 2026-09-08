#include "IR/MyOps.h"
#define GET_OP_CLASSES
#include <ranges>

#include "../key.h"
#include "IR/MyOps.cpp.inc"

#include "IR/MyDialect.h"
#include "llvm/Support/ScopedPrinter.h"

namespace my {
    void MyDialect::registerOps() {
        addOperations<
#define GET_OP_LIST
#include "IR/MyOps.cpp.inc"
        >();
    }

    mlir::LogicalResult SoftmaxOp::verify() {
        const auto returnType = mlir::cast<MyTensorType>(this->getType());
        if (const auto axis = getAxis();
            axis < 0 || axis >= returnType.getShape().size())
            return mlir::failure();
        return mlir::success();
    }

    ::mlir::LogicalResult SoftmaxOp::applyDistribution(::my::DistributeParallelAttrInterface disAttr) {
        const auto dataDisAttr = mlir::cast_or_null<my::DataParallelAttrInterface>(disAttr);
        if (!dataDisAttr) {
            return mlir::failure();
        }
        // 根据dpattr使用buff_cast,拆分然后新建多个op处理，最后使用buffer_cast合并
        const auto loc = this->getLoc();
        const int64_t axis = this->getAxis();
        const auto operandType = mlir::cast<MyTensorType>(this->getOperand().getType());
        const auto batch_dim_size = operandType.getShape()[0];
        const auto op = getOperation();
        auto builder = mlir::OpBuilder(op);
        builder.setInsertionPointAfter(op);
        // 先创建相关的op
        llvm::SmallVector<int64_t> tempShape(operandType.getShape());
        mlir::SmallVector<mlir::Type> midTypes;
        for (const auto [index,deviceId]: llvm::enumerate(dataDisAttr.getDeviceIds())) {
            auto part_batch_dim = batch_dim_size / dataDisAttr.getDP();
            // 处理不能均分的部分
            if (batch_dim_size % dataDisAttr.getDP() > index) {
                part_batch_dim++;
            }
            tempShape[0] = part_batch_dim;
            midTypes.push_back(MyTensorType::get(getContext(), tempShape, operandType.getElementType(), deviceId));
        }
        // 分开-》计算-》合并的op创建
        auto scatter = BufferCast::create(builder, loc, midTypes, getOperand(), dataDisAttr);
        llvm::SmallVector<mlir::Operation *> repeat_ops;
        llvm::SmallVector<mlir::Value> repeatValues;
        for (int i = 0; i < dataDisAttr.getDP(); i++) {
            auto disOp = SoftmaxOp::create(builder, loc, scatter.getResult(i), axis);
            repeat_ops.push_back(disOp);
            repeatValues.push_back(disOp.getResult());
        }
        auto gather = BufferCast::create(builder, loc, {this->getType()}, repeatValues, dataDisAttr);
        // 开始处理op之间的联系
        replaceAllUsesWith(gather.getOperation()->getResult(0));

        return mlir::success();
    }

    bool SoftmaxOp::supportsDistribution(::my::DistributeParallelAttrInterface disAttr) {
        return mlir::isa<my::DataParallelAttrInterface>(disAttr);
    }

    namespace {
        /// 构建ops中所有操作数引用外部value的记录，和使用它的op和参数索引
        llvm::MapVector<mlir::Value, llvm::SmallVector<std::pair<mlir::Operation *, int> > > getInputs(
            mlir::ArrayRef<mlir::Operation *> ops) {
            llvm::SetVector<mlir::Operation *> op_set(ops.begin(), ops.end());
            llvm::MapVector<mlir::Value, llvm::SmallVector<std::pair<mlir::Operation *, int> > > inputs;
            for (auto op: ops) {
                for (auto [operandIdx,operand]: llvm::enumerate(op->getOperands())) {
                    if (mlir::isa<mlir::BlockArgument>(operand)) {
                        // MapVector 会在value不存在时，自动调用无参构造函数进行创建
                        inputs[operand].push_back({nullptr, 0});
                    } else if (!op_set.contains(operand.getDefiningOp())) {
                        inputs[operand].push_back({op, operandIdx});
                    }
                }
            }
            return inputs;
        }

        /// 构建ops中所有结果被外部使用的记录，和创建其的op和结果索引
        llvm::MapVector<mlir::OpResult, std::pair<mlir::Operation *, int> > getOutputs(
            mlir::ArrayRef<mlir::Operation *> ops) {
            llvm::SetVector<mlir::Operation *> op_set(ops.begin(), ops.end());
            llvm::MapVector<mlir::OpResult, std::pair<mlir::Operation *, int> > outputs;
            for (auto op: ops) {
                for (auto [resIdx,res]: llvm::enumerate(op->getResults())) {
                    for (const auto user: res.getUsers()) {
                        if (op_set.contains(user)) continue;
                        outputs[res] = {op, resIdx};
                        break;
                    }
                }
            }
            return outputs;
        }

        /// 根据ops进行生成
        llvm::SmallString<4> getFusionName(mlir::ArrayRef<mlir::Operation *> ops) {
            llvm::SmallString<4> name;
            for (const auto op: ops) {
                name.append(op->getName().stripDialect());
                name.append("_");
                for (const auto operand_type: op->getOperandTypes()) {
                    if (auto shaped = mlir::cast_or_null<mlir::ShapedType>(operand_type)) {
                        for (const auto index: llvm::index_range(0, shaped.getRank())) {
                            if (shaped.isDynamicDim(index)) {
                                name.append("d_");
                            } else {
                                name.append(llvm::to_string(shaped.getDimSize(index)));
                                name.append("_");
                            }
                        }
                    }
                }
                name.append("X");
                for (const auto result_type: op->getResultTypes()) {
                    if (auto shaped = mlir::cast_or_null<mlir::ShapedType>(result_type)) {
                        for (const auto index: llvm::index_range(0, shaped.getRank())) {
                            if (shaped.isDynamicDim(index)) {
                                name.append("d_");
                            } else {
                                name.append(llvm::to_string(shaped.getDimSize(index)));
                                name.append("_");
                            }
                        }
                    }
                }
            }
            return name;
        }
    }

    mlir::LogicalResult DeviceKernelOp::FusionOps(mlir::RewriterBase &rewriter,
                                                  mlir::ArrayRef<mlir::Operation *> ops,
                                                  mlir::Location loc) {
        int64_t deviceId;
        if (const auto t = mlir::cast_or_null<my::MyTensorType>(ops.front()->getResultTypes().front())) {
            deviceId = t.getDeviceId();
        } else {
            return mlir::failure();
        }
        ops.front()->getOperands();
        const auto name = getFusionName(ops);
        const auto inputs_map = getInputs(ops);
        const auto outputs_map = getOutputs(ops);
        auto input_val = mlir::SmallVector<mlir::Value>();
        auto output_types = mlir::SmallVector<mlir::Type>();
        auto output_val = mlir::SmallVector<mlir::Value>();
        for (const auto &operand: inputs_map | std::views::keys) {
            input_val.push_back(operand);
        }
        for (const auto [result,_]: outputs_map) {
            output_types.push_back(result.getType());
        }
        // 创建并完善op内的region
        // 并构建了外部进入到deviceKernelOp的value连接
        auto deviceOp = DeviceKernelOp::create(rewriter, loc, output_types, name, deviceId, input_val);
        mlir::Block *block = &deviceOp.getRegion().emplaceBlock();
        // 在原始op和新建op之间的映射
        llvm::MapVector<mlir::Operation *, mlir::Operation *> op_map;
        // 复制op
        for (auto op: ops) {
            auto new_op = op->clone();
            op_map[op] = new_op;
            block->push_back(new_op);
            // 修改移动后的操作数的连接关系
            for (auto [operandIdx,operand]: llvm::enumerate(op->getOperands())) {
                if (mlir::isa<mlir::BlockArgument>(operand)) { continue; }
                if (op_map.contains(operand.getDefiningOp())) {
                    op_map[op]->setOperand(operandIdx,
                                           op_map[operand.getDefiningOp()]->getResult(
                                               llvm::cast<mlir::OpResult>(operand).getResultNumber()));
                }
            }
        }
        // 处理inputs和outputs的映射情况
        for (auto [index,input_map]: llvm::enumerate(inputs_map)) {
            auto operand = input_map.first;
            auto opAndOperandIdxs = input_map.second;
            auto arg = block->addArgument(operand.getType(), loc);
            for (auto [op,operandIdx]: opAndOperandIdxs) {
                op_map[op]->setOperand(operandIdx, arg);
            }
        }
        for (auto [result,opAndResultIdx]: outputs_map) {
            // 根据旧的op获取到新的op,然后根据resultIdx从新op中获取返回值的连接
            output_val.push_back(op_map[opAndResultIdx.first]->getResult(opAndResultIdx.second));
        }
        const auto savedInsertPointer = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToEnd(block);
        ReturnOp::create(rewriter, loc, output_val);
        rewriter.restoreInsertionPoint(savedInsertPointer);
        // 最后处理block结束之后，传出result和外部的联系
        for (const auto [idx,resAndOpAndResIdx]: llvm::enumerate(outputs_map)) {
            rewriter.replaceAllUsesWith(resAndOpAndResIdx.first, deviceOp->getResult(idx));
        }
        for (auto op: llvm::reverse(ops)) {
            rewriter.eraseOp(op);
        }
        return mlir::success();
    }
}

namespace my {
    llvm::LogicalResult BufferCast::canonicalize(my::BufferCast op, mlir::PatternRewriter &rewriter) {
        // 因为当前的Op没有实现Pure标记，但是实际该Op是无副作用的，
        // 所以当Op成为deadcode时，主动删除一下。
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return mlir::success();
        }
        // 之针对 scatter 的情况，这样只有一个操作数，便于判断。如果之前是buffercast,且输入输出的参数和dpattr都一样就消除当前的op,
        // 如果之前的cast也是deadcode,那么顺道也删除了。
        if (op.getOperands().size() != 1) {
            return mlir::failure();
        }
        const auto operand = op.getOperand(0);
        if (mlir::isa<mlir::BlockArgument>(operand)) {
            return mlir::failure();
        }
        if (!mlir::isa<BufferCast>(operand.getDefiningOp())) {
            return mlir::failure();
        }
        auto above_cast_op = mlir::cast<BufferCast>(operand.getDefiningOp());
        // 判断最终的输入输出参数是否等价
        const auto above_cast_op_operands = above_cast_op.getOperands();
        const auto op_results = op.getResults();
        // 判断参数数量
        if (above_cast_op_operands.size() != op_results.size()) {
            return mlir::failure();
        }
        // 判断参数类型
        for (int i = 0; i < above_cast_op_operands.size(); i++) {
            if (above_cast_op_operands[i].getType() != op_results[i].getType()) {
                return mlir::failure();
            }
        }
        // // 判断disAttr是否相等
        // if (op.getDisAttr() != above_cast_op.getDisAttr()) {
        // }

        // 执行消除
        rewriter.replaceOp(op, above_cast_op_operands);
        // 不使用下面的方式，是因为规范化方法被调用的pass可以在任意Op上，
        // 所以为了确保pass一定不会违反线程安全的约束，所以只使用replaceOp替换当前op，而不删除above_cast。
        // Greedy canonicalizer 会在受影响的op上再执行一次canonicalization。
        // 所以 above_cast 会再触发一次规范化方法，然后就会在规范化方法开头判断是否是 deadcode 进行移除。
        //
        // rewriter.replaceAllUsesWith(op_results, above_cast_op_operands);
        // rewriter.eraseOp(op);
        // if (above_cast_op.use_empty()) {
        //     rewriter.eraseOp(above_cast_op);
        // }
        return mlir::success();
    }
}
