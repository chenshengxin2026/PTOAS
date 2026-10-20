# Native WSE Calendar operations for A3 simulation

`include/PTO/IR/WSEOps.td` adds two experimental dialect operations. Both have
opaque read/write memory effects: communication, status, and epoch changes must
survive canonicalization and common-subexpression elimination.

| Operation | Operands | Static attributes |
| --- | --- | --- |
| `pto.wse_local` | lhs, rhs, output, logical-to-physical mapping | kind, rows, cols, inner, workers |
| `pto.tcalendar` | send, receive, routes, mapping, members, control, status | rows, cols, group_size, plan_slot, recv_offset, workers, nodes, opcode, red_op |

The local kinds are `gemm`, `silu`, `mul`, and `cast`. GEMM accepts FP16 inputs
and FP32 output; vector compute accepts FP32, with FP16 output for cast.
Calendar accepts FP16 data, i64 route words, and i32 binding/status arrays.
Only `ALLGATHER=1` and `NONE=0` are implemented. Verifiers reject unsupported
targets, types, group partitions, dimensions, row alignment, overlapping UB
regions, and capacity overflow.

Lowering in `PTOToEmitC/SyncComm/WSE.cpp` emits calls to:

```cpp
pto::wse::LocalGemm<Rows, Cols, Workers, Inner>(...);
pto::wse::LocalSilu<Rows, Cols, Workers>(...);
pto::wse::LocalMul<Rows, Cols, Workers>(...);
pto::wse::LocalCast<Rows, Cols, Workers>(...);
pto::wse::CalendarKernel<Rows, Cols, GroupSize, PlanSlot, RecvOffset, Workers, Nodes>(...);
```

Modules using these operations include `pto/wse/a3.hpp` in generated C++.
These are composite PTO-ISA helpers for the simulation backend; lowering does
not invent native WSE machine opcodes or bypass the normal PTOAS parser and
verifier. Physical PG, route bits, mapping values, and slot images remain
runtime binding operands, so deployment changes do not require recompilation.

`nodes=48` selects the 96-bit route profile; `nodes=40` selects the older
80-bit profile. Both use two 64-bit scalar loads per 16-byte route entry and
store expected remote bytes at offset 12. The schema records the profile so
deployment cannot silently mix them.

Build `PTOASPythonPackage` with Python bindings enabled, then run:

```bash
build/tools/ptoas/ptoas input.pto --pto-arch=a3 -o output.cpp
python -m pytest -q --noconftest -c /dev/null test/wse/test_calendar.py
```

The tests invoke the checkout's built native CLI. They check valid lowering,
preservation of repeated stateful calls, and negative geometry/type/opcode
contracts. Full A3 tests compile emitted C++ against the modified PTO-ISA
headers and compare the original FFN with a separate CPU golden.
