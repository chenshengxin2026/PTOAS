# WSE Calendar lowering on A3

The `wse-sim-1026` branch adds the internal `pto.tcalendar` operation for the
[PyPTO FFN example](https://github.com/chenshengxin2026/pypto/blob/wse-sim-1026/docs/en/dev/wse-a3.md).
The linked guide contains the four-branch build and real A3 validation entry.

The operation takes a launch-owned `i64` context, send/receive Tile buffers,
an `i32` expected remote-byte count, an axis and ordered logical members.
It has read/write memory effects because communication remains observable
even if the receive tile is unused. Lowering emits
`pto::wse::RunCalendar<axis, members...>` and includes `pto/wse/calendar.hpp`.
The synchronous ISA facade owns entry/exit barriers and transport completion.

Verification requires `--pto-arch=a3 --pto-level=level3`, static FP16 rank-2
row-major, unboxed L1 tiles, full valid dimensions, 32-byte source rows,
1–24 distinct nonnegative int32 IDs and nonoverlapping allocated L1 ranges.
Receive geometry must equal concatenation along the selected axis and stay
within 128 KiB. The expected count is `(members - 1) * source_valid_bytes`;
the local slice consumes capacity but does not advance remote completion.
Dynamic, partial or unallocated geometry is rejected.

```bash
python test/wse_calendar.py --ptoas build/tools/ptoas/ptoas
```

The test compiles a valid operation and rejects wrong byte counts, duplicate
members, aliasing, partial valid rows, wrong memory space and unsupported
architecture. `test/lit/pto/wse_calendar.pto` also checks emitted C++.
Native A3 execution uses the matching PTO-ISA CCE mock and simpler deployment;
PTOAS itself does not synthesize mapping, routes, slots or epochs.
