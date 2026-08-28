#include "IR/MyOps.h"
#define GET_OP_CLASSES
#include "../key.h"
#include "IR/MyOps.cpp.inc"

#include "IR/MyDialect.h"

namespace my {
    void MyDialect::registerOps() {
        addOperations<
#define GET_OP_LIST
#include "IR/MyOps.cpp.inc"
        >();
    }

    mlir::LogicalResult GetTensorOp::verify() {
        const auto buffer = this->getBuffer();
        const auto deviceId = this->getDeviceId();
        // 是外部传入参数，只检查deviceId
        if (mlir::isa<mlir::BlockArgument>(buffer)) {
            for (const auto deviceIds = mlir::cast<BufferType>(buffer.getType()).getDevices();
                 const auto it: deviceIds) {
                if (it == deviceId) {
                    return mlir::success();
                }
            }
            return mlir::failure();
        }
        // 否则检查deviceId和类型
        const auto return_tensor_type = this->getType();
        auto bufferOp = mlir::cast_or_null<BufferOp>(buffer.getDefiningOp());
        // 目前应该只有bufferOp可以创建该类型。
        if (!bufferOp) {
            return mlir::failure();
        }
        for (const auto it: bufferOp.getTensors()) {
            const auto tensor_type = mlir::cast_or_null<MyTensorType>(it.getType());
            if (!tensor_type) {
                return mlir::failure();
            }
            if (tensor_type.getDeviceId() == deviceId) {
                if (tensor_type != return_tensor_type) {
                    return mlir::failure();
                }
                return mlir::success();
            }
        }
        return mlir::failure();
    }

    mlir::LogicalResult BufferOp::verify() {
        const auto tensors = getTensors();
        for (const auto devices = mlir::cast<BufferType>(getType()).getDevices();
             auto [index, device_id, tensor]: llvm::enumerate(devices, tensors)) {
            if (auto tensor_type = cast_or_null<MyTensorType>(tensor.getType());
                device_id != tensor_type.getDeviceId()) {
                return llvm::failure();
            }
        }
        return mlir::success();
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
        auto op = getOperation();
        mlir::OpBuilder builder = mlir::OpBuilder(op);
        builder.setInsertionPointAfter(op);
        // 因为后面会移除这个op,所以先同样复制一份,替换。
        // TODO: 暂时什么也不做,之后再重写转化。
        auto nop = SoftmaxOp::create(builder, this->getLoc(), this->getInput(), this->getAxis());
        nop->setAttr(my::KDeviceFunc, mlir::StringAttr::get(getContext(), "todo"));
        replaceAllUsesWith(nop.getResult());

        return mlir::success();
    }

    bool SoftmaxOp::supportsDistribution(::my::DistributeParallelAttrInterface disAttr) {
        return mlir::isa<my::DataParallelAttrInterface>(disAttr);
    }
}
