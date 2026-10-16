// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "../PTOToEmitCEmitters.h"
#include "SyncCommInternal.h"

using namespace mlir;

namespace mlir::pto {
struct CalendarToEmitC : OpConversionPattern<TCalendarOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(TCalendarOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        auto params = rewriter.getArrayAttr(
            {op.getRowsAttr(), op.getColsAttr(), op.getGroupSizeAttr(), op.getPlanSlotAttr(), op.getRecvOffsetAttr(),
             op.getWorkersAttr(), op.getNodesAttr()});
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{}, "pto::wse::CalendarKernel", ArrayAttr{}, params, adaptor.getOperands());
        return success();
    }
};

struct WSELocalToEmitC : OpConversionPattern<WSELocalOp> {
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(WSELocalOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
    {
        std::string callee;
        SmallVector<Attribute> params{op.getRowsAttr(), op.getColsAttr(), op.getWorkersAttr()};
        if (op.getKind() == "gemm") {
            callee = "pto::wse::LocalGemm";
            params.push_back(op.getInnerAttr());
        } else if (op.getKind() == "silu")
            callee = "pto::wse::LocalSilu";
        else if (op.getKind() == "mul")
            callee = "pto::wse::LocalMul";
        else
            callee = "pto::wse::LocalCast";
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{}, callee, ArrayAttr{}, rewriter.getArrayAttr(params), adaptor.getOperands());
        return success();
    }
};

void populateWSEPatterns(RewritePatternSet& patterns, TypeConverter& converter, MLIRContext* ctx)
{
    patterns.add<CalendarToEmitC, WSELocalToEmitC>(converter, ctx);
}
} // namespace mlir::pto
