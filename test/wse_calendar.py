# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Focused CLI regression: python test/wse_calendar.py --ptoas build/tools/ptoas/ptoas."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptoas", type=Path, required=True)
    args = parser.parse_args()
    source = (Path(__file__).parent / "lit/pto/wse_calendar.pto").read_text()
    cases = [
        ("valid", source, "a3", "pto::wse::RunCalendar<1, 9, 2>"),
        (
            "bytes",
            source.replace("constant 64 : i32", "constant 128 : i32"),
            "a3",
            "expected bytes",
        ),
        (
            "members",
            source.replace("array<i64: 9, 2>", "array<i64: 9, 9>"),
            "a3",
            "distinct",
        ),
        (
            "alias",
            source.replace("constant 4096 : i64", "constant 32 : i64"),
            "a3",
            "must not overlap",
        ),
        (
            "valid_rows",
            source.replace("constant 2 : index", "constant 1 : index"),
            "a3",
            "full valid",
        ),
        ("space", source.replace("loc=mat", "loc=vec"), "a3", "FP16 L1 tiles"),
        ("target", source, "a5", "requires --pto-arch=a3"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        for name, text, arch, expected in cases:
            path = Path(directory) / f"{name}.pto"
            path.write_text(text)
            result = subprocess.run(
                [
                    str(args.ptoas.resolve()),
                    f"--pto-arch={arch}",
                    "--pto-level=level3",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            diagnostic = result.stdout + result.stderr
            assert (result.returncode == 0) == (name == "valid"), diagnostic
            assert expected in diagnostic, diagnostic
            if name == "valid":
                assert "pto/wse/calendar.hpp" in diagnostic
            print(f"PASS: {name}")


if __name__ == "__main__":
    main()
