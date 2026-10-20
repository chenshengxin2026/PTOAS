// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

static bool wsePtrElement(mlir::Value value, mlir::Type expected)
{
    return cast<mlir::pto::PtrType>(value.getType()).getElementType() == expected;
}

LogicalResult mlir::pto::TCalendarOp::verify()
{
    auto* ctx = getContext();
    if (!isTargetArchA3(getOperation()))
        return emitOpError("WSE simulation currently requires the A3 target");
    if (getOpcode() != 1 || getRedOp() != 0)
        return emitOpError("A3 Calendar mock supports ALLGATHER=1, NONE=0 only");
    if (getRows() < 1 || getRows() > 16 || getCols() < 16 || getCols() % 16 || getCols() > 4096 || getGroupSize() < 1 ||
        getGroupSize() > 32 || getWorkers() < 1 || getWorkers() > 32 || getWorkers() % getGroupSize())
        return emitOpError("invalid static geometry, 32-byte alignment, or group partition");
    if ((getNodes() != 40 && getNodes() != 48) || getPlanSlotAttr().getInt() < 0 || getRecvOffset() > 192 * 1024 ||
        getRecvOffset() < getRows() * getCols() * 2 || getRecvOffset() % 32 ||
        getRecvOffset() + getRows() * getCols() * getGroupSize() * 2 > 192 * 1024)
        return emitOpError("invalid route profile or overlapping/out-of-capacity A3 UB arena");
    if (!wsePtrElement(getSend(), Float16Type::get(ctx)) || !wsePtrElement(getRecv(), Float16Type::get(ctx)) ||
        !wsePtrElement(getRoutes(), IntegerType::get(ctx, 64)))
        return emitOpError("requires f16 send/recv and i64 route words");
    for (Value value : {getMapping(), getMembers(), getControl(), getStatus()})
        if (!wsePtrElement(value, IntegerType::get(ctx, 32)))
            return emitOpError("binding and status operands must be i32 pointers");
    return success();
}

LogicalResult mlir::pto::WSELocalOp::verify()
{
    auto* ctx = getContext();
    if (!isTargetArchA3(getOperation()))
        return emitOpError("WSE simulation currently requires the A3 target");
    auto kind = getKind();
    if (getRows() < 1 || getRows() > 16 || getCols() < 16 || getCols() % 16 || getCols() > 4096 || getWorkers() < 1 ||
        getWorkers() > 32 || !wsePtrElement(getMapping(), IntegerType::get(ctx, 32)))
        return emitOpError("invalid local geometry or mapping type");
    if (kind != "gemm" && kind != "silu" && kind != "mul" && kind != "cast")
        return emitOpError("unknown WSE local operation");
    if (kind == "gemm" && (getInner() < 64 || getInner() % 64 || getInner() > 65536))
        return emitOpError("GEMM K must be a positive multiple of 64, at most 65536");
    Type input = kind == "gemm" ? Type(Float16Type::get(ctx)) : Type(Float32Type::get(ctx));
    Type output = kind == "cast" ? Type(Float16Type::get(ctx)) : Type(Float32Type::get(ctx));
    if (!wsePtrElement(getLhs(), input) || !wsePtrElement(getRhs(), input) || !wsePtrElement(getOut(), output))
        return emitOpError("unsupported input/output dtype for local operation");
    if (kind != "gemm" && getRows() * getCols() * (kind == "cast" ? 6 : 16) > 192 * 1024)
        return emitOpError("local vector temporaries exceed A3 UB");
    return success();
}
