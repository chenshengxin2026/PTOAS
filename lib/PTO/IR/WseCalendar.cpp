// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult TCalendarOp::verify()
{
    auto src = cast<TileBufType>(getSrc().getType());
    auto dst = cast<TileBufType>(getDst().getType());
    auto members = getMembers();
    int64_t memberCount = members.size();
    if (members.empty() || members.size() > 24)
        return emitOpError("requires 1..24 resident group members on A3");
    llvm::SmallDenseSet<int64_t> ids;
    for (auto id : members)
        if (id < 0 || id > INT32_MAX || !ids.insert(id).second)
            return emitOpError("requires distinct nonnegative int32 logical IDs");
    if (getAxis() != 0 && getAxis() != 1)
        return emitOpError("requires axis 0 or 1");
    int64_t addresses[2];
    unsigned tileIndex = 0;
    for (Value value : {getSrc(), getDst()}) {
        auto tile = cast<TileBufType>(value.getType());
        auto space = dyn_cast<AddressSpaceAttr>(tile.getMemorySpace());
        if (!space || space.getAddressSpace() != AddressSpace::MAT || !tile.getElementType().isF16() ||
            tile.getShape().size() != 2 || tile.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor) ||
            tile.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox))
            return emitOpError("requires flat row-major FP16 L1 tiles");
        for (auto dim : tile.getShape())
            if (dim <= 0 || dim > 65536)
                return emitOpError("requires positive static tile dimensions within 128 KiB");
        auto allocation = value.getDefiningOp<AllocTileOp>();
        APInt address;
        if (!allocation || !allocation.getAddr() || !matchPattern(allocation.getAddr(), m_ConstantInt(&address)))
            return emitOpError("requires statically addressed L1 allocations (level3)");
        addresses[tileIndex++] = address.getSExtValue();
        for (unsigned dim = 0; dim < 2; ++dim) {
            int64_t valid = tile.getValidShape()[dim];
            if (ShapedType::isDynamic(valid)) {
                Value extent = dim == 0 ? allocation.getValidRow() : allocation.getValidCol();
                APInt constant;
                if (!extent || !matchPattern(extent, m_ConstantInt(&constant)))
                    return emitOpError("requires static full valid tiles");
                valid = constant.getSExtValue();
            }
            if (valid != tile.getShape()[dim])
                return emitOpError("requires packed full valid tiles");
        }
    }
    auto shape = src.getShape();
    if (shape[1] % 16 != 0)
        return emitOpError("requires 32-byte aligned source rows");
    for (int axis = 0; axis < 2; ++axis)
        if (dst.getShape()[axis] != shape[axis] * (static_cast<uint64_t>(axis) == getAxis() ? memberCount : 1))
            return emitOpError("destination shape does not match ordered AllGather");
    if (dst.getShape()[0] * dst.getShape()[1] * 2 > 128 * 1024)
        return emitOpError("receive capacity exceeds 128 KiB");
    APInt expected;
    if (!matchPattern(getExpected(), m_ConstantInt(&expected)) ||
        expected.getSExtValue() != (memberCount - 1) * shape[0] * shape[1] * 2)
        return emitOpError("expected bytes must equal remote valid payload, excluding self");
    int64_t sendBytes = shape[0] * shape[1] * 2;
    int64_t recvBytes = dst.getShape()[0] * dst.getShape()[1] * 2;
    for (unsigned i = 0; i < 2; ++i)
        if (addresses[i] < 0 || addresses[i] % 32 || addresses[i] > 512 * 1024 - (i == 0 ? sendBytes : recvBytes))
            return emitOpError("requires aligned allocations within A3 L1 capacity");
    if (addresses[0] < addresses[1] + recvBytes && addresses[1] < addresses[0] + sendBytes)
        return emitOpError("send and receive storage must not overlap");
    return success();
}
