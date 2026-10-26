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
        SmallVector<Attribute> params{op.getAxisAttr()};
        for (auto member : op.getMembers())
            params.push_back(rewriter.getI64IntegerAttr(member));
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{}, "pto::wse::RunCalendar", ArrayAttr{}, rewriter.getArrayAttr(params),
            adaptor.getOperands());
        return success();
    }
};
void populateWseCalendarPatterns(RewritePatternSet& patterns, TypeConverter& converter, MLIRContext* ctx)
{
    patterns.add<CalendarToEmitC>(converter, ctx);
}
} // namespace mlir::pto
