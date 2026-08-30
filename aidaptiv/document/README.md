# aiDAPTIV Library

C++ library for **Mixture-of-Experts (MoE) offload** and **KV-cache reuse**: expert weights can be moved to SSD to reduce peak memory use, and KV caches can be saved to DRAM/SSD so repeated prefixes skip re-computation.

## Contents

- Overview
- Integrating aiDAPTIV
- MoE and KV configuration
- Demo
- APIs
- Supported Phison firmware

## Overview

**MoE offload** stores expert weights on SSD and loads them on demand, using configurable VRAM and DRAM cache budgets so you can trade disk latency for a smaller memory footprint.

**KV-cache reuse** saves prompt (and tokens generated) KV caches under an offload directory. On the next run with the same prefix, `restore_kv_cache()` refills the in-memory KV cache so `llama_decode()` can skip those tokens.

## Integrating aiDAPTIV

Typical order: create `Aidaptiv` → load model → create context → `init()` → inference (optionally restore / save KV) → flush and cleanup.

MoE expert offload is set on `llama_model_params`. KV-cache offload and reuse are set on `aidaptiv::setup_params` and used through `Aidaptiv` APIs.

### 1. Create and initialize `Aidaptiv`

Set paths and KV options **before** loading the model. The offload directory must already exist and be writable.

```cpp
const char * model_path   = "E:\\gpt-oss-20b-Q4_K_M.gguf";
const char * offload_path = "D:\\";
std::string  debug_log_path = offload_path;  // or e.g. "D:\\log"

aidaptiv::setup_params sp;
sp.model_path             = model_path;
sp.mmproj_model_path      = "";              // set if using a multimodal model
sp.ssd_kv_offload_gb      = -1;             // SSD KV budget (GB); -1 = auto (recommended), 0 = off
sp.dram_kv_offload_gb     = -1;             // DRAM KV budget (GB); -1 = auto (recommended), 0 = off
sp.kv_cache_resume_policy = 0;               // 0 = no resume and delete old caches, 1 = try resume on startup
sp.flash_attn             = true;

aidaptiv::Aidaptiv adptv(offload_path, debug_log_path, sp);
```


| `setup_params` field                       | Description                                                                            |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| `model_path`                               | GGUF model path. Required so saved KV caches match this model on restore.              |
| `mmproj_model_path`                        | Multimodal projector path; leave empty for text-only.                                  |
| `ssd_kv_offload_gb` / `dram_kv_offload_gb` | KV offload budgets (GB). Recommended: `-1` (auto). `0` disables.                      |
| `kv_cache_resume_policy`                   | `0` = delete old caches; `1` = try to resume cached KV caches when the runtime starts. |
| `flash_attn`                               | Flash attention flag; applied when you call `init()`.                                  |


Use `adptv.offload_path()` for all later `offload_folder` settings (it may differ from the path you passed in on some deployments).

### 2. Set model parameters and load the model


| Parameter                | Description                                                              |
| ------------------------ | ------------------------------------------------------------------------ |
| `offload_folder`         | Directory for offload data. Use `adptv.offload_path().c_str()`.          |
| `vram_experts_cached_gb` | VRAM budget (GB) for cached experts. `0` disables MoE offload features.  |
| `dram_experts_cached_gb` | DRAM budget (GB) for cached experts. `0` disables DRAM-side MoE offload. |


```cpp
llama_model_params model_params = llama_model_default_params();
model_params.offload_folder         = adptv.offload_path().c_str();
model_params.vram_experts_cached_gb = 3;   // example: 3 GB VRAM for MoE experts
model_params.dram_experts_cached_gb = 0;   // 0 = disable DRAM-side MoE offload

llama_model * model = llama_model_load_from_file(model_path, model_params);
```

### 3. Create context and call `init()`

After `llama_init_from_model()`, call `init()` once to attach KV-cache support to your context. It uses `model_path` / `mmproj_model_path` from `setup_params`.

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx   = 4096;   // must cover prompt + generation
ctx_params.n_batch = 512;

llama_context * ctx = llama_init_from_model(model, ctx_params);

std::vector<llama_adapter_lora *> lora_init;   // lora adapters already loaded in llama.cpp
std::vector<aidaptiv::lora_info>  lora_adapters;
adptv.init(ctx, model, lora_init, lora_adapters);
```

### 4. Inference with KV restore and save

**Restore** (before the first `llama_decode()`): try to load a matching prefix from cache, then build the first batch from the tokens that were **not** restored.

```cpp
// always decode the last token to get fresh logits for sampling.
std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
auto stats = adptv.restore_kv_cache(lookup_tokens, {}, {}, /*slot_id*/ 0, /*hit_in_device*/ 0);
// hit_in_device: how many leading tokens already have KV in GPU; use 0 if none

uint32_t total_reuse = stats.dram_reuse_token_cnt + stats.ssd_reuse_token_cnt;

uint32_t skip = total_reuse;

llama_batch batch = llama_batch_get_one(
    prompt_tokens.data() + skip,
    (int32_t) prompt_tokens.size() - (int32_t) skip);

// ... llama_decode / sample loop ...
```

**Important — adjust the batch after restore:**

| Rule | Why |
| ---- | --- |
| Pass `lookup_tokens` (prompt minus last token) to `restore_kv_cache()` | The last prompt token must always be decoded to get fresh logits for sampling.  |
| Set `skip = dram_reuse_token_cnt + ssd_reuse_token_cnt` | Those leading tokens already have KV in the context; do not pass them to `llama_decode()` again. |
| Start the decode loop with `n_pos = skip` (or equivalent) | Keeps position tracking aligned with how many tokens are already in the KV cache. |

**Save** (after generation): persist tokens whose KV caches are in the context, then flush to disk.

```cpp
adptv.save_kv_cache(tokens_to_save, {}, {});
adptv.flush_kv_cache();
```

For text-only prompts without LoRA, pass `{}` for `mtmd_info` and `lora_adapters` (as in the example above).

### 5. Reclaim offload space (`reclaim_offload_space`)

When you need disk space under `offload_path`, call `reclaim_offload_space()`.

```cpp
// Reclaim at least 1 GiB
std::size_t freed = adptv.reclaim_offload_space(1024LL * 1024 * 1024);
```

`expect_size` is the **minimum number of bytes you want to free**. Eviction order:

1. **Unused expert caches** under `temp_cache` (oldest first; in-use folders skipped).
2. If still short, **unused KV-cache sets** (`kv_cache` / `prefix_tree` / `model_info` per model), also oldest first and skipping in-use files.

The return value is how many bytes were actually removed.


| Situation                                                                  | Result                                                       |
| -------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Enough unused expert and/or KV cache                                       | Deletes the chosen set(s); typically `freed >= expect_size`. |
| Every cache is in use                                                      | No deletion; returns `0`.                                    |
| Unused caches exist but their total size is still below `expect_size`      | No deletion; returns `0`.                                    |


Caches that are currently in use are not evicted. KV subfolders locked via the lock folder API are also kept (see **KV cache lock folders**).

### SSD space cleanup

When SSD space is short, cleanup runs in these cases:

1. **KV cache not enough** (need room to save / extend KV)
   1. Auto-delete for the required size.
   2. Or call `reclaim_offload_space(expect_size)` yourself to reclaim a specific amount.
2. **Expert cache not enough** (need room to store MoE expert weights)
   1. Auto-delete for the required size.
   2. Or call `reclaim_offload_space(expect_size)` yourself to reclaim a specific amount.
3. **Delete order** (shared by both auto paths): unused expert data first, then unused KV cache. In-use data is never deleted. Oldest unused data is removed first; cleanup stops once enough space is freed.

### Node-aligned access

aiDAPTIV reads and writes KV only in fixed-size **nodes**. Both save and restore operate on complete nodes.

For standard attention-only and ISWA models, only complete nodes are used. If the token count is not aligned to the node size, the trailing tokens that do not fill a complete node are discarded and are **not saved or restored**.

| Model type                              | Node size    | Unaligned behavior |
| --------------------------------------- | ------------ | ------------------ |
| Standard attention-only / ISWA          | **8** tokens | Use complete nodes; discard the trailing unaligned tokens. |
| Recurrent / hybrid (e.g. Qwen3.5)       | **128** tokens | Save/restore only when the requested token count is aligned to the node size. |

### Recurrent and hybrid models (e.g. Qwen3.5)

Recurrent/hybrid models keep a **compressed state** that advances through the sequence. Because the compressed state is node-based, the token count you pass to `save_kv_cache()` must be **divisible by the node size** (after you trim the last token if it was not decoded yet, as in the demo). If it is not aligned to the node size, **the cache is not saved**. Use the same node size on restore.

**Reuse only where you saved:** On the next run, `restore_kv_cache()` skips leading tokens only up to the length you passed to a successful `save_kv_cache()` earlier. Tokens beyond that save point were never written to disk, so they are always recomputed. You also cannot “reuse less” than that save (e.g. skip only half of it)—to reuse from an earlier position, you need a separate `save_kv_cache()` at that position.

**Prompt longer than one node**

If the full prompt is **longer** than the node size but its length is **not** a multiple of the node size, you cannot save the whole prompt in one call. Instead:

1. **Prefill the first aligned chunk** — Run prefill up to the largest complete-node boundary.
2. **Save** — Call `save_kv_cache()` with that aligned token count.
3. **Continue prefill** — Prefill the rest of the prompt.
4. **Optional: save again during decode** — When the total storable length reaches the next node boundary after generation, call `save_kv_cache()` again.

Every `save_kv_cache()` call for recurrent/hybrid models must use a token count that is a multiple of the node size. You can save a checkpoint in the middle of prefill—you do not have to wait until the entire prompt is in the context.

**Example (Qwen3.5):**


| Situation                                      | What to do                                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Prompt is exactly **one node**                 | Prefill the full prompt → save once.                                                                        |
| Prompt is **two full nodes plus extra tokens** | Prefill the first **two nodes** → save → prefill the remaining tokens → decode → save again at the next node boundary. |
| Prompt is exactly **two nodes**                | Prefill the full prompt → save once.                                                                        |
| Prompt has extra tokens, save all at once      | **Not saved** (the token count is not a multiple of the node size).                                         |


### Multimodal prompts (`mtmd_info`)

KV cache keys include multimodal content, not only text tokens. Use `aidaptiv::mtmd_chunk_info` and `aidaptiv::mtmd_seq_info` when the prompt contains images or other media.

```cpp
struct mtmd_chunk_info {
    uint64_t    img_chunk_id = 0;   // image chunk id (e.g. from mtmd_input_chunk_get_id)
    std::size_t token_cnt    = 0;   // how many KV / decode tokens that image expands to after mtmd encoding
};
using mtmd_seq_info = std::vector<mtmd_chunk_info>;  // one entry per image chunk, in prompt order
```

**Compact token list for lookup / save / restore:** pass **one** `-1` (`MULTIMODAL_TOKEN`) per image in the token sequence. Do **not** expand the image into `token_cnt` copies of `-1` in `lookup_tokens` / `tokens_to_save`. The real patch count lives only in `mtmd_info[].token_cnt`; the library expands internally when hashing and restoring KV.


| Field          | Meaning                                                                                                                                                                                                                                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `img_chunk_id` | Stable id for the image chunk (same value at save and restore). **Recommended:** hash the raw image (or audio) bytes into a `uint64_t` so the same file always gets the same id. **llama-server** does this with an FNV-style hash of the bitmap, stores it on the chunk id, then passes `std::stoull(mtmd_input_chunk_get_id(...))` into `mtmd_info`. |
| `token_cnt`    | How many tokens that image expands to after `mtmd` encoding (patch / image-token count). Used with the single `-1` placeholder in the compact token list.                                                                                                                                                                                              |


**How to build it in your app**

1. Set `setup_params.mmproj_model_path` for multimodal models.
2. For each image chunk in prompt order: set `img_chunk_id` (from `mtmd_input_chunk_get_id`, or your own hash of the image/path) and `token_cnt` (from mtmd encoding).
3. In the token list passed to `restore_kv_cache()` / `save_kv_cache()`, insert **one** placeholder token `-1` (`MULTIMODAL_TOKEN`) per image, in the same order as `mtmd_info`.
4. Pass the **same** `mtmd_info` to `restore_kv_cache()` and `save_kv_cache()`. For restore, use `lookup_tokens` (prompt minus last token), not the full `prompt_tokens`.

```cpp
// Compact tokens: one MULTIMODAL_TOKEN (-1) per image, not token_cnt copies.
// Example layout: [text..., -1, text..., -1, text...]
aidaptiv::mtmd_seq_info mtmd_info = {
    { img_chunk_id_0, n_tokens_image_0 },  // n_tokens_image_* = expanded count after mtmd
    { img_chunk_id_1, n_tokens_image_1 },
};

std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
auto stats = adptv.restore_kv_cache(lookup_tokens, mtmd_info, lora_adapters, slot_id, hit_in_device);
// ... decode ...
adptv.save_kv_cache(tokens_to_save, mtmd_info, lora_adapters, slot_id);
```

If `img_chunk_id`, `token_cnt`, or compact placeholder layout differ between save and restore, prefix lookup will miss or match the wrong cache.

### LoRA adapters (`lora_info`)

When LoRA is enabled, the KV-cache key includes the adapters in use (in addition to `prompt_tokens` and `mtmd_info`). Describe each adapter with `aidaptiv::lora_info`:

```cpp
struct lora_info {
    std::string path;            // LoRA GGUF path (hashed into the KV-cache key)
    float       scale = 0.0f;    // must match the scale applied at inference
    std::string task_name;       // optional; not part of the KV key today
    std::string prompt_prefix;   // optional; not part of the KV key today
};
```


| Field                        | Meaning                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `path`                       | Path to the LoRA weights file. Same path at save and restore.                      |
| `scale`                      | Adapter strength; must match what you pass to llama.cpp when applying the adapter. |
| `task_name`, `prompt_prefix` | Optional metadata; safe to leave empty for KV reuse.                               |


**How to use** (same pattern as llama.cpp `common` + llama-server)

1. Fill `lora_adapters` with each adapter’s `path` and `scale`.
2. After `llama_model_load_from_file()`, call `llama_adapter_lora_init(model, path)` for each entry and collect pointers in `lora_init`.
3. After `llama_init_from_model()`, **apply** the adapters to the context with `llama_set_adapters_lora()` (or `common_set_adapter_lora()` from `common/common.h`) using the same scales as in `lora_adapters`.
4. Call `adptv.init(ctx, model, lora_init, lora_adapters)`.
5. Pass the **same** `lora_adapters` to `restore_kv_cache()` and `save_kv_cache()`.

```cpp
const char * lora_path = "path/to/adapter.gguf";
const float  lora_scale = 1.0f;

std::vector<aidaptiv::lora_info> lora_adapters = {
    { lora_path, lora_scale, "", "" },
};

llama_adapter_lora * lora = llama_adapter_lora_init(model, lora_path);
std::vector<llama_adapter_lora *> lora_init = { lora };

llama_context * ctx = llama_init_from_model(model, ctx_params);

// Apply LoRA to the context (required before decode; same as common_init / server)
llama_set_adapters_lora(ctx, lora_init.data(), lora_init.size(), &lora_scale);

adptv.init(ctx, model, lora_init, lora_adapters);

std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
auto stats = adptv.restore_kv_cache(lookup_tokens, {}, lora_adapters, slot_id, hit_in_device);
// ... decode ...
adptv.save_kv_cache(tokens_to_save, {}, lora_adapters, slot_id);
```

If `path`, `scale`, or the set of adapters differ between save and restore, prefix lookup will miss. Without LoRA, pass `{}` for `lora_adapters` everywhere.

**Note:** In llama.cpp `common`, `task_name` and `prompt_prefix` are read from the LoRA GGUF metadata at load time. For aiDAPTIV KV lookup only `path` and `scale` matter; you may leave the strings empty when filling `lora_info` yourself.

### Draft models (speculative decoding / MTP)

In speculative decoding the draft context decodes the same positions as the target, so it has **its own KV cache**. That cache is not covered by the target's `Aidaptiv`: create a **second instance** for the draft context and drive it alongside the target.

Skip the second instance when the draft shares the target's memory (Gemma 4 assistant-style drafts, where `llama_get_ctx_other(ctx_dft) == ctx_tgt`). Those cells are restored with the target.

```cpp
aidaptiv::setup_params sp_dft = sp;                       // same KV budgets as the target
sp_dft.model_path = draft_path.empty() ? sp.model_path : draft_path;   // target GGUF if the MTP head is in it
sp_dft.flash_attn = llama_flash_attn(ctx_dft);

aidaptiv::Aidaptiv adptv_dft(adptv.offload_path(), debug_log_path, sp_dft);
adptv_dft.init(ctx_dft, model_dft);
```

Each instance keys its caches on its own `model_path`, so the draft files stay separate from the target's even when the draft reuses the target GGUF.

**Restore both to the same prefix.** The draft must hold KV for exactly the tokens the target skips; a shorter draft prefix makes it attend over a hole and every prediction is rejected. The two instances evict independently, so restore the draft first and cap the target with what the draft actually returned:

```cpp
std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);

auto dft_stats = adptv_dft.restore_kv_cache(lookup_tokens, {}, {}, 0, 0);
uint32_t dft_reuse = dft_stats.dram_reuse_token_cnt + dft_stats.ssd_reuse_token_cnt;
lookup_tokens.resize(std::min<size_t>(lookup_tokens.size(), dft_reuse));

auto stats = adptv.restore_kv_cache(lookup_tokens, {}, {}, 0, 0);
uint32_t skip = stats.dram_reuse_token_cnt + stats.ssd_reuse_token_cnt;

// the prefill rewrites the draft positions from `skip` on; drop the cells left there
if (dft_reuse > skip) {
    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, (llama_pos) skip, -1);
}
```

**Save and flush both** with the same tokens; the draft mirrors the target positions, so one token sequence keys both caches:

```cpp
adptv.save_kv_cache(tokens_to_save, {}, lora_adapters);
adptv_dft.save_kv_cache(tokens_to_save, {}, {});

adptv.flush_kv_cache();
adptv_dft.flush_kv_cache();
```

**Note:** on recurrent / hybrid targets, save the draft at the same node-aligned checkpoints as the target (see **Recurrent and hybrid models**). If the target saves a checkpoint the draft did not, the next run's draft prefix is shorter and caps the target's reuse.

### 6. Shut down cleanly

Flush pending KV writes, then remove expert temp caches under `offload_path`:

| API | Behavior |
| --- | --- |
| `remove_owned_temp_caches()` | Remove only the expert temp caches this run published or adopted. Skip caches still held by another live process (keeps other runs' caches for resume). **Demo uses this.** |
| `remove_temp_caches()` | Remove every not being used expert temp cache under the offload root. |

```cpp
adptv.flush_kv_cache();

// Prefer when other processes may still need caches under offload_path (demo uses this):
adptv.remove_owned_temp_caches();

// Or wipe every not being used expert temp cache under the offload root:
// adptv.remove_temp_caches();
```

## Demo

Requirements:

- MSVC Compiler
- CMake

### Build

Open a terminal:

```cmd
cmake -B build -S . -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DLLAMA_BUILD_BORINGSSL=ON -DGGML_BACKEND_DL=ON
cmake --build build --config Release --target aidaptiv-demo llama-common
```

### Run

#### `ada`

Offload features require the **ada** service. Run setup from an **elevated (Administrator)** terminal.

- Location in the package: `aidaptiv\ada`
- Copy next to **aidaptiv-demo.exe**: `ada.exe`, `wService_create.bat`, and `wService_delete.bat`

**Start (create / activate service):**

```cmd
wService_create.bat
```

**Stop (remove service):**

```cmd
wService_delete.bat
```

#### `aidaptiv-demo.exe` CLI

Remember to install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

CMD
```cmd
aidaptiv-demo.exe -m <model> -o <offload_path> [-n n_predict]
    [-lora adapter.gguf] [-ls scale]
    [-vc vram_experts_cached_gb] [-dc dram_experts_cached_gb]
    [-sk ssd_kv_offload_gb] [-dk dram_kv_offload_gb] 
    [-kr kv_cache_resume_policy] [-fa]
    [-mtp] [-md draft.gguf] [-dn n_draft] [prompt]
```

**Example (MoE + KV cache):**

```cmd
aidaptiv-demo.exe -m gpt-oss-20b-Q4_K_M.gguf -n 128 -o D:\\ -vc 3 -sk -1 -dk -1
```

Run the same command twice with the same prompt: the second run should print non-zero reuse counts from `restore_kv_cache()`.

**Example (LoRA + KV cache, two-run reuse test):**

```cmd
REM Pass 1: build cache
aidaptiv-demo.exe -m base.gguf -o D:\\ -lora adapter.gguf -ls 1.0 -sk -1 -dk -1 -kr 1 -n 16

REM Pass 2: should print non-zero Dram/Cache reuse
aidaptiv-demo.exe -m base.gguf -o D:\\ -lora adapter.gguf -ls 1.0 -sk -1 -dk -1 -kr 1 -n 16
```

**Example (MTP draft + KV cache):**

```cmd
REM Qwen 3.5: the MTP head comes from the target GGUF
aidaptiv-demo.exe -m qwen3.5-35b.gguf -o D:\\ -sk -1 -dk -1 -kr 1 -mtp -n 64

REM Gemma 4: separate assistant model
aidaptiv-demo.exe -m gemma4-26b.gguf -o D:\\ -sk -1 -dk -1 -kr 1 -mtp -md gemma4-assistant.gguf -n 64
```

The demo keeps a second `Aidaptiv` instance for the draft context (see **Draft models**), so the second run also prints `MTP ... reuse token` counts. A Gemma 4 assistant shares the target's memory instead, and is restored with the target.

**Flags:**


| Flag / argument | Meaning                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `-m`            | Path to the GGUF model.                                                        |
| `-n`            | Number of tokens to generate.                                                  |
| `-o`            | Offload data directory (required).                                             |
| `-vc`           | VRAM budget (GB) for cached experts; `0` disables MoE offload feature.         |
| `-dc`           | DRAM budget (GB) for cached experts; `0` disables DRAM-side MoE offload.       |
| `-sk`           | SSD budget (GB) for KV-cache offload. Recommended: `-1` (auto). `0` disables.  |
| `-dk`           | DRAM budget (GB) for KV-cache offload. Recommended: `-1` (auto). `0` disables. |
| `-kr`           | KV resume policy: `0` = delete kv caches, `1` = try resume on startup.         |
| `-fa`           | Enable flash attention.                                                        |
| `-lora`         | Path to a LoRA adapter GGUF (optional).                                        |
| `-ls`           | LoRA scale when `-lora` is set (default: `1.0`).                               |
| `-mtp`          | Enable speculative decoding with an MTP draft (see below).                     |
| `-md`           | Draft model GGUF; omit to take the MTP head from the target GGUF.              |
| `-dn`           | Draft tokens per step when `-mtp` is set (default: `3`).                       |
| trailing args   | Prompt text (default: `Hello my name is`).                                     |


**Recommended MoE offload (`-vc`)**

The table below lists suggested values for `vram_experts_cached_gb` (`-vc`) by GPU VRAM. Lower `-vc` if you run out of GPU memory (OOM), or raise it if you have spare VRAM. For `-dc` (DRAM expert cache), pick a value from your system RAM budget (use `0` to turn off DRAM-side expert caching).


| GPU VRAM | Recommended `-vc` | Example models                            |
| -------- | ----------------- | ----------------------------------------- |
| 16 GB    | `4`               | Gemma 4 26B, Qwen 3.5 35B                 |
| 32 GB    | `20`              | GPT-OSS 120B, Qwen 3 80B                  |
| 64 GB    | `45`              | Nemotron 120B, GLM-4.5-Air, Qwen 3.5 122B |


**Note:** Use a **MoE model** if you want to exercise MoE offload (`-vc` / `-dc`). KV-cache reuse works with any model when `-o` and KV budgets are set.

## APIs

Cache hit rates are in **[0, 1]**:

```cpp
// Single-token (n_tokens == 1): VRAM expert cache hit rate.
float llama_moe_get_single_token_vram_hit_rate(struct llama_context * ctx);

// Single-token (n_tokens == 1): DRAM expert cache hit rate.
float llama_moe_get_single_token_dram_hit_rate(struct llama_context * ctx);

// Multi-token (n_tokens > 1): VRAM expert cache hit rate.
float llama_moe_get_multi_token_vram_hit_rate(struct llama_context * ctx);

// Multi-token (n_tokens > 1): DRAM expert cache hit rate.
float llama_moe_get_multi_token_dram_hit_rate(struct llama_context * ctx);

// Combined VRAM expert cache hit rate.
float llama_moe_get_vram_hit_rate(struct llama_context * ctx);

// Combined DRAM expert cache hit rate.
float llama_moe_get_dram_hit_rate(struct llama_context * ctx);

// Reset all hit-rate counters.
void llama_moe_reset_hit_rate(struct llama_context * ctx);
```

### KV cache lock folders

You can **lock** KV caches so they are not evicted during cleanup, while still allowing `restore_kv_cache()` to reuse them.

- **Lock folders** — subfolders you mark as locked. KV caches here are not evicted. You can add data manually or save with `save_kv_cache(..., subfolder)`.
- **Dynamic KV cache** — entries in the default location (not in a dedicated lock subfolder). These may be evicted when space is needed.

**Unlock** a folder before you delete or move it on disk. Unlocked folders are not reused or offloaded to; the runtime releases resources tied to them.


| Task                  | API                                                                       |
| --------------------- | ------------------------------------------------------------------------- |
| Get lock status       | `get_lock_folder_map()`                                                   |
| Lock / unlock         | `update_lock_folders(set_lock_map)`                                       |
| Save into a subfolder | `save_kv_cache(tokens, {}, {}, seq_id, "my_session")` then lock if needed |


```cpp
// Save KV into a named subfolder (create/populate the folder), then lock it
adptv.save_kv_cache(tokens_to_save, {}, {}, 0, "my_session");

// Get lock status: subfolder name -> "lock" or "unlock"
auto locks = adptv.get_lock_folder_map();
for (const auto & [name, state] : locks) {
    // name: subfolder under offload_path; state: lock status string
}

// Lock or unlock existing subfolders under inference
std::unordered_map<std::string, bool> set_lock_map = {
    { "my_session", true },   // lock (folder must already exist)
    { "old_run",    false },  // unlock - release resources; unlock before delete/move
};
std::string err = adptv.update_lock_folders(set_lock_map);  // empty if OK
```

## Supported Phison firmware

Supported firmware version **prefix**:

- EVFZ

### Lookup Methods

**Powershell Tools**

```powershell
Get-PhysicalDisk | Select DeviceId, FirmwareVersion
```

```powershell
PS C:\WINDOWS\system32> Get-PhysicalDisk | Select DeviceId, FirmwareVersion

DeviceId FirmwareVersion
-------- ---------------
0        EVFZ00.1         --> Phison SSD
1        SHFM30.1         --> Non-Phison SSD
```

**NVMe Command**

You also can use NVMe Identify Command to get the SSD firmware version.

Reference Link: [NVME_IDENTIFY_CONTROLLER_DATA](https://learn.microsoft.com/zh-tw/windows/win32/api/nvme/ns-nvme-nvme_identify_controller_data)

**Windows Management Instrumentation (WMI)**

```powershell
Get-WmiObject -Class Win32_DiskDrive | Select-Object Model, FirmwareRevision
```

```powershell
PS C:\Users\phison> Get-WmiObject -Class Win32_DiskDrive | Select-Object Model, FirmwareRevision

Model    FirmwareRevision
-----    ----------------
PCIe SSD EVFZ00.1         --> Phison SSD
SATA SSD SHFM30.1         --> Non-Phison SSD
```

