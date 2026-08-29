#include "Transforms/MyPasses.h"

#include <IR/MyOps.h>


#include "IR/MyAttrs.h"
#include "IR/MyDialect.h"
#include "IR/MyOpInterfaces.h"

#include "../key.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace my {
#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETERSPASS
#include "Transforms/MyPasses.h.inc"

    struct MarkDistributeParallelParametersPass
            : impl::MarkDistributeParallelParametersPassBase<
                MarkDistributeParallelParametersPass> {
        using MarkDistributeParallelParametersPassBase<
                    MarkDistributeParallelParametersPass>::
                MarkDistributeParallelParametersPassBase;

    protected:
        void runOnOperation() override {
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
            auto module = getOperation();
            LLVM_DEBUG(
                llvm::dbgs() << llvm::formatv("root op: {0}\n", module->getName()));
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("DPNums: {0}\n", DPNums));
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("TPNums: {0}\n", TPNums));
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("EPNums: {0}\n", EPNums));

            if (TPNums != 1) {
                llvm::errs() << "TPNums not supported currently!\n";
                signalPassFailure();
                return;
            }
            if (DPNums != 1) {
                std::vector<int64_t> deviceIds;
                for (int i = 0; i < DPNums; i++) {
                    deviceIds.push_back(i);
                }
                auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums, deviceIds);
                module->walk(
                    [&dp_attr](mlir::func::FuncOp op) { op->setAttr(KDPAttrName, dp_attr); });
            }
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
        }
    };
}

namespace my {
#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Transforms/MyPasses.h.inc"

    struct ApplyDistributeTransformPass : impl::ApplyDistributeTransformPassBase<ApplyDistributeTransformPass> {
        using ApplyDistributeTransformPassBase<
                    ApplyDistributeTransformPass>::
                ApplyDistributeTransformPassBase;

    protected:
        void runOnOperation() override {
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
            auto func = getOperation();
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("root op: {0}\n", func->getName()));
            auto dp_attr = llvm::dyn_cast_or_null<my::DistributeParallelAttrInterface>(
                func->getAttr(KDPAttrName));
            if (!dp_attr)
                llvm_unreachable("error!");
            func->walk([&](mlir::Operation *op) {
                if (auto dis_op = llvm::dyn_cast_or_null<my::DistributeParallelOpInterface>(op)) {
                    if (dis_op.supportsDistribution(dp_attr)) {
                        if (dis_op.applyDistribution(dp_attr).succeeded()) {
                            LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                                "Apply DataParallelism to {0}\n", op->getName()));
                            op->erase();
                        }
                    }
                }
            });
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
        }
    };
}

namespace my {
    // namespace {
    struct BufferCastOpDeviceRegionFusion : mlir::OpRewritePattern<::my::BufferCast> {
        // 使用using 自动继承父类的构造函数
        using mlir::OpRewritePattern<::my::BufferCast>::OpRewritePattern;

        void addops(llvm::SetVector<mlir::Operation *> &ops, mlir::Operation *op) const {
            if (!mlir::isa<my::DistributeParallelOpInterface>(op)) return;
            ops.insert(op);
            for (auto user: op->getUsers()) {
                addops(ops, user);
            }
        }

        virtual mlir::LogicalResult matchAndRewrite(BufferCast op, mlir::PatternRewriter &rewriter) const override {
            // match判断
            // 1. 是scatter
            // 2. 后继不是devicekernelop
            if (op.getOperands().size() != 1) return mlir::failure();
            for (auto user: op.getResult(0).getUsers()) {
                if (mlir::isa<my::DeviceKernelOp>(user)) {
                    return mlir::failure();
                }
            }

            auto loc = op->getLoc();
            llvm::SmallVector<llvm::SetVector<mlir::Operation *> > op_list;
            for (auto res: op->getResults()) {
                rewriter.setInsertionPointAfterValue(res);
                llvm::SetVector<mlir::Operation *> ops;
                for (auto use: res.getUsers()) {
                    addops(ops, use);
                }
                if (!ops.empty()) op_list.push_back(ops);
            }
            if (op_list.empty()) return llvm::failure();
            for (auto ops: op_list) {
                if (!my::DeviceKernelOp::FusionOps(rewriter, ops.takeVector(), loc)
                    .succeeded()) {
                    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("fusion error!"));
                    return llvm::failure();
                };
            }
            return llvm::success();
        }
    };

    // }

#define GEN_PASS_DEF_DEVICEREGIONFUSIONPASS
#include "Transforms/MyPasses.h.inc"

    struct DeviceRegionFusionPass : impl::DeviceRegionFusionPassBase<DeviceRegionFusionPass> {
    protected:
        void runOnOperation() override {
            mlir::RewritePatternSet pat_set(&getContext());
            pat_set.addWithLabel<BufferCastOpDeviceRegionFusion>(
                mlir::StringRef("BufferCastOpDeviceRegionFusion"), pat_set.getContext(), 100);
            if (mlir::applyPatternsGreedily(getOperation(),
                                            mlir::FrozenRewritePatternSet(std::move(pat_set))).failed()) {
                signalPassFailure();
            }
        }
    };
}
