# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise the native dialect verifier and EmitC lowering through the built CLI."""

import subprocess
from pathlib import Path

import pytest

PTOAS = Path(__file__).resolve().parents[2] / "build/tools/ptoas/ptoas"
CALENDAR_OP = """"pto.tcalendar"(%s, %d, %r, %m, %g, %c, %status)
  {rows = 8 : i64, cols = 96 : i64, group_size = 8 : i64,
   plan_slot = 0 : i64, recv_offset = 8192 : i64, workers = 32 : i64,
   nodes = 48 : i64, opcode = 1 : i64, red_op = 0 : i64}
  : (!pto.ptr<f16>, !pto.ptr<f16>, !pto.ptr<i64>, !pto.ptr<i32>,
     !pto.ptr<i32>, !pto.ptr<i32>, !pto.ptr<i32>) -> ()"""
CALENDAR = """module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @calendar(%s: !pto.ptr<f16>, %d: !pto.ptr<f16>, %r: !pto.ptr<i64>,
    %m: !pto.ptr<i32>, %g: !pto.ptr<i32>, %c: !pto.ptr<i32>, %status: !pto.ptr<i32>) attributes {pto.kernel} {
    OPERATION
    return
  }
}"""
GEMM = """module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<cube>} {
  func.func @gemm(%a: !pto.ptr<f16>, %b: !pto.ptr<f16>, %c: !pto.ptr<f32>, %m: !pto.ptr<i32>) attributes {pto.kernel} {
    "pto.wse_local"(%a, %b, %c, %m) {kind = "gemm", rows = 8 : i64, cols = 96 : i64,
      inner = 7168 : i64, workers = 32 : i64}
      : (!pto.ptr<f16>, !pto.ptr<f16>, !pto.ptr<f32>, !pto.ptr<i32>) -> ()
    return
  }
}"""


def compile_ir(tmp_path, source, arch="a3"):
    src, cpp = tmp_path / "input.pto", tmp_path / "output.cpp"
    src.write_text(source)
    result = subprocess.run(
        [str(PTOAS), str(src), f"--pto-arch={arch}", "-o", str(cpp)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, cpp.read_text() if cpp.exists() else ""


def test_calendar_lowering_and_memory_effects(tmp_path):
    result, cpp = compile_ir(
        tmp_path, CALENDAR.replace("OPERATION", CALENDAR_OP + "\n" + CALENDAR_OP)
    )
    assert result.returncode == 0, result.stderr
    assert '#include "pto/wse/a3.hpp"' in cpp
    assert cpp.count("pto::wse::CalendarKernel<") == 2, (
        "CSE must preserve both stateful Calendar calls"
    )


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("rows = 8", "rows = 0", "geometry"),
        ("group_size = 8", "group_size = 3", "partition"),
        ("cols = 96", "cols = 97", "alignment"),
        ("recv_offset = 8192", "recv_offset = 32", "overlapping"),
        ("recv_offset = 8192", "recv_offset = 196608", "capacity"),
        ("nodes = 48", "nodes = 41", "profile"),
        ("opcode = 1", "opcode = 2", "ALLGATHER"),
        ("red_op = 0", "red_op = 1", "NONE"),
        ("plan_slot = 0", "plan_slot = -1", "profile"),
        ("!pto.ptr<f16>", "!pto.ptr<f32>", "f16"),
    ],
)
def test_calendar_rejects_invalid_contract(tmp_path, old, new, message):
    source = CALENDAR.replace("OPERATION", CALENDAR_OP).replace(old, new)
    result, _ = compile_ir(tmp_path, source)
    assert result.returncode != 0 and message in result.stderr, result.stderr


def test_local_gemm_lowering(tmp_path):
    result, cpp = compile_ir(tmp_path, GEMM)
    assert result.returncode == 0, result.stderr
    assert "pto::wse::LocalGemm<8, 96, 32, 7168>" in cpp


@pytest.mark.parametrize(
    "old,new,message",
    [
        ('kind = "gemm"', 'kind = "unknown"', "unknown WSE"),
        ("inner = 7168", "inner = 65", "multiple of 64"),
        ("!pto.ptr<f32>", "!pto.ptr<f16>", "dtype"),
    ],
)
def test_local_gemm_rejects_invalid_contract(tmp_path, old, new, message):
    result, _ = compile_ir(tmp_path, GEMM.replace(old, new))
    assert result.returncode != 0 and message in result.stderr, result.stderr


@pytest.mark.parametrize("source", [GEMM, CALENDAR.replace("OPERATION", CALENDAR_OP)])
def test_requires_a3_target(tmp_path, source):
    result, _ = compile_ir(tmp_path, source.replace('"a3"', '"a5"'), arch="a5")
    assert result.returncode != 0 and "requires the A3 target" in result.stderr, (
        result.stderr
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
