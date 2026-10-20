# WSE mock: native PTOAS operations and lowering

This branch adds native compiler support for the WSE Calendar A3 simulation.
The upstream baseline is `8f454ed0cfcbef701967851dc5ec84ecfb136fb3`.
The PR targets the fork's `wse-mock` branch; review code is on
`wse-mock-calendar-a3`.

## Changes and interfaces

| File | Role |
| ---- | ---- |
| [WSEOps.td](../../include/PTO/IR/WSEOps.td) | `pto.wse_local` and `pto.tcalendar` definitions and memory effects |
| [WSEVerification.cpp](../../lib/PTO/IR/PTOCore/WSEVerification.cpp) | A3 target, element type, geometry, alignment, group and capacity checks |
| [WSE.cpp](../../lib/PTO/Transforms/PTOToEmitC/SyncComm/WSE.cpp) | EmitC lowering to `pto::wse` helpers |
| [test_calendar.py](../../test/wse/test_calendar.py) | 17 native CLI tests for lowering, rejection and stateful-call preservation |

`pto.wse_local` supports GEMM, SiLU, multiplication and cast.
`pto.tcalendar` carries send/receive tensors and runtime route, mapping,
membership, control and status operands. Geometry is static; PG and physical
placement remain runtime inputs. Opaque read/write effects prevent CSE from
discarding repeated stateful calls.

Generated C++ includes `pto/wse/a3.hpp` and calls `LocalGemm`, `LocalSilu`,
`LocalMul`, `LocalCast` or `CalendarKernel`. These are composite A3 helpers
implementing the mock backend, without introducing a real WSE machine opcode.
See the [operation reference](../wse-calendar-a3.md) for signatures and ABI.

## Build and run

Build the checkout following the root README using compatible LLVM/MLIR 19
and Python bindings enabled. Build target `PTOASPythonPackage` supplies the
native CLI used by the tests.

```bash
build/tools/ptoas/ptoas input.pto --pto-arch=a3 -o output.cpp
python -m pytest -q --noconftest -c /dev/null test/wse/test_calendar.py
```

`input.pto` is a generated `op_*.pto` from the modified PyPTO frontend.
Use the matching `wse-mock-calendar-a3` branches of
[PyPTO](https://github.com/chenshengxin2026/pypto/tree/wse-mock-calendar-a3),
[PTO-ISA](https://github.com/chenshengxin2026/pto-isa/tree/wse-mock-calendar-a3)
and [simpler](https://github.com/chenshengxin2026/simpler/tree/wse-mock-calendar-a3).
After all PRs merge, use their `wse-mock` branches together.
The [cross-repository run guide](https://github.com/chenshengxin2026/simpler/blob/wse-mock-calendar-a3/docs/wse-mock/README.md)
builds a full FFN, compiles each emitted global kernel as a separate translation
unit with bisheng, and compares actual A3 output against CPU golden.

## Validation and limits

On 2026-10-15, the four-layer stack passed 46 targeted CPU/PTOAS tests,
the separate C++ receiver contract, and 21 FFN executions on Ascend910_9392.
The original `T=8,H=7168,I=3072,P=32` case passed two seeds, three mappings,
and two repeats. Maximum absolute error was `0.015625`; maximum relative RMS
was `8.2343e-5`, against pointwise `atol=0.002, rtol=0.01` and RMS `<0.002`.
All workers' two gather snapshots and same-seed mapping/repeat outputs were
bit-exact.

Supported operations are static 2D compute and AllGather on A3.
The 48-node/96-bit and 40-node/80-bit route profiles are explicit simulation
ABIs. The compiler validates modeled WSE/A3 resource constraints; mapping,
NoC planning, PG, Batcher and CCE hardware mechanisms remain stub/mock layers.
No real WSE fabric timing or performance is validated.
