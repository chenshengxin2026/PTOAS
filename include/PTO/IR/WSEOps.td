// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


// Experimental, static-shape WSE operations for the A3 simulation backend.
// The opaque read/write effects include transport and private epoch state.
def TCalendarOp : PTO_Op<"tcalendar", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "WSE Calendar AllGather with deployment-bound transport";
  let arguments = (ins PtrType:$send, PtrType:$recv, PtrType:$routes,
    PtrType:$mapping, PtrType:$members, PtrType:$control, PtrType:$status,
    I64Attr:$rows, I64Attr:$cols, I64Attr:$group_size, I64Attr:$plan_slot,
    I64Attr:$recv_offset, I64Attr:$workers, I64Attr:$nodes,
    I64Attr:$opcode, I64Attr:$red_op);
  let hasVerifier = 1;
}

def WSELocalOp : PTO_Op<"wse_local", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Static local WSE computation on A3 PTO tiles";
  let arguments = (ins PtrType:$lhs, PtrType:$rhs, PtrType:$out,
    PtrType:$mapping, StrAttr:$kind, I64Attr:$rows, I64Attr:$cols,
    I64Attr:$inner, I64Attr:$workers);
  let hasVerifier = 1;
}
