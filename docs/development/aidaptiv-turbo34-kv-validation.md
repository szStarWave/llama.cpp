# aiDAPTIV Turbo3/4 KV Validation

## Scope

This document validates the low-impact Turbo3/4 KV cache integration on branch
`codex/integrate-turbo34-kv`.

- llama.cpp base: `ed3ff7d5283e60c0d3c5bb5bfe1113686d8f8360`
- TurboQuant source: `2f2f32f5d9517518c9e860f30131acb09840a965`
- aiDAPTIV SDK/DLL: 3.0.7.0
- Test date: 2026-08-10
- Host: Windows amd64, AMD Radeon(TM) 8060S Graphics, wave64
- Turbo2, TQ weights, InnerQ, and non-Vulkan GPU backends are out of scope.

The recommended runtime combinations are `K=q8_0,V=turbo3` and
`K=q8_0,V=turbo4`. Explicit user K/V choices are never rewritten.

## Source Mapping

The integration was manually reduced to the required Turbo3/4 KV paths. The
following TurboQuant commits were used as references:

| Commit | Reference area |
| --- | --- |
| `00fda770b` | GGML type IDs, packed blocks, and CPU codec |
| `2a716ac47` | Vulkan dequantization, SET_ROWS, FA, and WHT shaders |
| `17d4422a4` | Codec quality test |
| `5bb743ca9` | AMD wave64 ballot handling |
| `b43efe599` | Turbo3 Vulkan pipeline registration |
| `e7ad3a9d9` | Backend operation coverage |
| `f58ee0e97` | Strided view initialization |
| `1f7946026` | llama graph and KV cache behavior reference |

No whole source file was copied from the b10320-based fork. The aiDAPTIV
dispatch ABI and llama memory interfaces were not changed.

## Build And Static Checks

### T34-BUILD-001: Release build

Command:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' `
  --build build-validation-b10068 --config Release -j 4
```

Expected: all Release targets build.

Actual: PASS.

### T34-BUILD-002: independent CTest set

Command:

```powershell
ctest -C Release --output-on-failure -j 4 `
  -E 'test-download-model|test-backend-ops|test-eval-callback|test-thread-safety|test-state-restore-fragmented|test-save-load-state'
```

Expected: all environment-independent tests pass.

Actual: PASS, 36/36. The excluded model tests depend on a zero-byte downloaded
`stories15M-q4_0.gguf`. The full Vulkan backend test has a pre-integration
baseline crash after about 304 seconds and is replaced by targeted Turbo tests
below.

### T34-BUILD-003: CLI and DLL ABI

Expected:

- `turbo3` and `turbo4` are accepted for target and draft K/V cache types.
- checkpoint defaults remain 8192/256.
- aiDAPTIV `--dry-run` completes without an ABI error.

Actual: PASS. `test-arg-parser` also passed.

## Codec And Backend Tests

### T34-CODEC-001: CPU codec

Command:

```powershell
.\build-validation-b10068\bin\Release\test-turbo-quant.exe
```

Actual:

| Type | MSE | Cosine | Result |
| --- | ---: | ---: | --- |
| Turbo3 basis | `1.0186e-9` | `1.000000` | PASS |
| Turbo4 fixed vector | `0.110608801` | `0.995648563` | PASS |

Turbo dequantization intentionally remains in the WHT-rotated domain, so the
generic float round-trip test skips Turbo3/4. The dedicated codec test applies
the inverse WHT.

### T34-VK-001: WHT and SET_ROWS

Coverage:

- forward and inverse `TURBO_WHT`
- Turbo3 and Turbo4 `SET_ROWS`
- I32 and I64 row indices
- single token, batch prefill, contiguous input, strided view, and large dispatch
- head dimensions 128 and 256

Actual: PASS, 50/50 on Vulkan0. No `0/0` result was accepted.

### T34-VK-002: Flash Attention

Command:

```powershell
.\build-validation-b10068\bin\Release\test-backend-ops.exe test `
  -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_[KV]=turbo' -j 1
```

Coverage:

- `q8_0/turbo3`, `q8_0/turbo4`
- `turbo3/turbo3`, `turbo4/turbo4`
- `turbo3/q8_0`, `turbo4/q8_0`
- head dimensions 128 and 256
- batch sizes 1 and 8

Actual: PASS, 16/16 on AMD wave64 Vulkan.

## Model And Memory Tests

### T34-MODEL-001: CPU K/V matrix

Model: Qwen2.5 1.5B Q4_K_M. All commands used `-dev none -ngl 0 -st`.

| K/V | Process result | Output quality |
| --- | --- | --- |
| `q8_0/turbo3` | PASS | valid |
| `q8_0/turbo4` | PASS | valid |
| `turbo3/turbo3` | PASS | materially degraded on this high-GQA model |
| `turbo4/turbo4` | PASS | materially degraded on this high-GQA model |
| `turbo3/q8_0` | PASS | materially degraded on this high-GQA model |
| `turbo4/q8_0` | PASS | materially degraded on this high-GQA model |

TurboQuant itself defaults to asymmetric K/V selection for large-GQA models.
This integration does not copy that policy because explicit user types must not
be changed. Herdsman should select `q8_0/turboN` by default and preserve explicit
overrides.

### T34-MEM-001: KV buffer allocation

Model: Qwen2.5 1.5B, context 4096.

| K/V | K MiB | V MiB | Total MiB |
| --- | ---: | ---: | ---: |
| `f16/f16` | 56.00 | 56.00 | 112.00 |
| `q8_0/q8_0` | 29.75 | 29.75 | 59.50 |
| `turbo3/turbo3` | 10.94 | 10.94 | 21.88 |
| `turbo4/turbo4` | 14.44 | 14.44 | 28.88 |
| `q8_0/turbo3` | 29.75 | 10.94 | 40.69 |
| `q8_0/turbo4` | 29.75 | 14.44 | 44.19 |

Per component, Turbo3 is 19.5% of f16 and 36.8% of q8_0. Turbo4 is
25.8% of f16 and 48.5% of q8_0. Both pass the required compression thresholds.

## aiDAPTIV Tests

### T34-ADP-001: global persistence namespace

Configuration: CPU server, Qwen2.5 1.5B, `K=q8_0,V=turbo3`, SSD KV enabled.

Expected:

- the first run does not restore an old f16/q8 or old Turbo layout;
- the second process restores the new Turbo codec revision;
- save, flush, restore, and `/shutdown` complete without DLL faults.

Actual: PASS. First run `cache_n=0`; restart `cache_n=160`.

### T34-ADP-002: managed folder namespace and lock lifecycle

Configuration: prefix `codext34b-` and a valid `doc-v1-<sha256>` cache ID.

Actual:

| Run | Folder state | Persistent hit | Final lock |
| --- | --- | ---: | --- |
| first | `existing=0` | 0 | unlocked |
| restart | `lock=1`, `existing=1` | 160 | unlocked |

PASS. A new managed folder now skips persistent restore until that folder has
been created. This prevents it from borrowing a matching global cache from a
different cache identity.

### T34-ADP-003: server API regressions

Using `LLAMA_TEST_MODEL_FILE` with the local Qwen2.5 model and
`N_GPU_LAYERS=0`, the focused pytest set passed 10/10:

- `/props` aiDAPTIV capability
- legal and illegal cache IDs
- missing server cache prefix
- `/shutdown`

The test utility now preserves an explicit zero GPU layer value and supports a
local model override when HTTPS is unavailable.

## Herdsman Application E2E

### T34-HERD-001: real GenerateContent call path

The E2E client was built from the current
`D:\code\herdsman-feature\super-agent` worktree. A Go overlay adds only the
external test file to the real `pkg/api/agui/models/llamacpp` package. Requests
therefore execute `llamacpp.Model.GenerateContent()` and cover Herdsman request
conversion, local image encoding, capability discovery, prompt preflight,
managed document cache selection, streaming, and response parsing.

The model was Qwen3.5 27B Q4_K_M with its BF16 mmproj. To protect the Codex host
from model-level Vulkan failures, the server used `--device none`,
`--gpu-layers 0`, and four CPU threads. Logs confirm 0/65 layers offloaded to
GPU. The production Vulkan model matrix remains external as described below.

Build and run commands:

```powershell
go test -c `
  -overlay=D:\code\llama.cpp-b9568-phison\llama.cpp\validation-logs\turbo34-integration-20260810\herdsman-overlay.json `
  -o D:\code\llama.cpp-b9568-phison\llama.cpp\validation-logs\turbo34-integration-20260810\herdsman-cpu\herdsman-llamacpp-e2e.test.exe `
  ./pkg/api/agui/models/llamacpp

python validation-logs\turbo34-integration-20260810\run-herdsman-turbo-cpu.py turbo3
python validation-logs\turbo34-integration-20260810\run-herdsman-turbo-cpu.py turbo4
```

Actual results:

| K/V | Phase | Text streaming | Document + image | Single image | Two-turn image history | Exit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `q8_0/turbo3` | first process | 896 | 1536 | 0, valid output | 1051 | client/server 0 |
| `q8_0/turbo3` | restart | 896 SSD | 1536 SSD | not repeated | not repeated | client/server 0 |
| `q8_0/turbo4` | first process | 896 | 1536 | 0, valid output | 1051 | client/server 0 |
| `q8_0/turbo4` | restart | 896 SSD | 1536 SSD | not repeated | not repeated | client/server 0 |

PASS. Both recommended K/V combinations produced valid output for every
scenario. Streaming emitted partial and final responses. The document preflight
created a 1536-token managed prefix, and the actual multimodal request reused
that prefix. On restart, the existing-folder build-only task was skipped, the
request restored 1536 tokens from SSD, and the final duplicate save was skipped.
The second image turn reused 1051 tokens from the same-process host context.

All four server processes exited with code 0 through `/shutdown`. No access
violation, assertion, NaN, or other configured fatal marker appeared. Raw client,
server, and combined result logs are under
`validation-logs/turbo34-integration-20260810/herdsman-cpu`.

## External Vulkan Model Matrix

Model-level Vulkan commands are not run from a Codex process. A driver timeout
or an interactive child can terminate the host application. Run these scripts
from a separate PowerShell after reviewing the arguments:

```powershell
.\validation-logs\turbo34-integration-20260810\run-vulkan-qwen25-matrix.ps1 `
  -ConfirmVulkanModelRun

.\validation-logs\turbo34-integration-20260810\run-vulkan-large-model-smoke.ps1 `
  -ConfirmVulkanModelRun
```

Both scripts use hidden `llama-server` processes, bounded health/request waits,
`/shutdown`, and a `finally` cleanup. They never leave `llama-cli` interactive.

The first script covers the full Qwen2.5 K/V matrix, records output and KV
allocation logs, and performs two-process aiDAPTIV checks for the recommended
combinations. The second covers Qwen3.5 27B hybrid vision, Qwen3.5 35B A3B with
expert and KV offload, and Gemma4 E2B vision for Turbo3 and Turbo4 V caches.

## Known Limits And Acceptance

- Recommended `q8_0/turbo3` and `q8_0/turbo4` output is valid and persistent.
- Turbo K on Qwen2.5 high GQA is a quality risk. It remains available because
  explicit user configuration is not rewritten.
- Non-128 model head dimensions are padded per head in the llama graph. The
  backend FA test uses storage-valid 128/256 shapes; model tests cover padding.
- Existing Qwen3.5 image KV persistence remains limited to the prefix before the
  first media chunk to avoid the known aiDAPTIV DLL access violation. Same-process
  host image context reuse remains available.
- The external large-model scripts must pass before a production release. GPU
  timeout, access violation, NaN, empty output, missing aiDAPTIV restart reuse,
  or failed compression thresholds are release blockers.
