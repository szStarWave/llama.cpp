#include "aidaptiv-dispatch.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Default stubs
namespace {

[[noreturn]] void aidaptiv_not_loaded(const char * fn) {
    LLAMA_LOG_ERROR("%s: aidaptiv_core is not loaded; cannot dispatch into it\n", fn);
    throw std::runtime_error("aidaptiv_core is not loaded");
}

size_t stub_kv_cache_get_cache_size(const llama_kv_cache &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("kv_cache_get_cache_size");
}
void stub_kv_cache_read(llama_kv_cache &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("kv_cache_read");
}
void stub_kv_cache_write(llama_kv_cache &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("kv_cache_write");
}
void stub_kv_cache_get_cached_positions(const llama_kv_cache &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("kv_cache_get_cached_positions");
}

uint32_t stub_mr_get_size(const llama_memory_recurrent &) {
    aidaptiv_not_loaded("mr_get_size");
}
size_t stub_mr_get_cache_size(const llama_memory_recurrent &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("mr_get_cache_size");
}
void stub_mr_kv_cache_read(const llama_memory_recurrent &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("mr_kv_cache_read");
}
void stub_mr_kv_cache_write(llama_memory_recurrent &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("mr_kv_cache_write");
}
void stub_mr_get_cached_positions(const llama_memory_recurrent &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("mr_get_cached_positions");
}

uint32_t stub_iswa_get_size(const llama_kv_cache_iswa &) {
    aidaptiv_not_loaded("iswa_get_size");
}
size_t stub_iswa_get_cache_size(const llama_kv_cache_iswa &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("iswa_get_cache_size");
}
void stub_iswa_kv_cache_read(llama_kv_cache_iswa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("iswa_kv_cache_read");
}
void stub_iswa_kv_cache_write(llama_kv_cache_iswa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("iswa_kv_cache_write");
}
void stub_iswa_get_cached_positions(const llama_kv_cache_iswa &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("iswa_get_cached_positions");
}

uint32_t stub_dsa_get_size(const llama_kv_cache_dsa &) {
    aidaptiv_not_loaded("dsa_get_size");
}
size_t stub_dsa_get_cache_size(const llama_kv_cache_dsa &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("dsa_get_cache_size");
}
void stub_dsa_kv_cache_read(llama_kv_cache_dsa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("dsa_kv_cache_read");
}
void stub_dsa_kv_cache_write(llama_kv_cache_dsa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("dsa_kv_cache_write");
}
void stub_dsa_get_cached_positions(const llama_kv_cache_dsa &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("dsa_get_cached_positions");
}

uint32_t stub_hyb_get_size(const llama_memory_hybrid &) {
    aidaptiv_not_loaded("hyb_get_size");
}
size_t stub_hyb_get_cache_size(const llama_memory_hybrid &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("hyb_get_cache_size");
}
void stub_hyb_kv_cache_read(llama_memory_hybrid &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("hyb_kv_cache_read");
}
void stub_hyb_kv_cache_write(llama_memory_hybrid &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("hyb_kv_cache_write");
}
void stub_hyb_get_cached_positions(const llama_memory_hybrid &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("hyb_get_cached_positions");
}

uint32_t stub_hyb_iswa_get_size(const llama_memory_hybrid_iswa &) {
    aidaptiv_not_loaded("hyb_iswa_get_size");
}
size_t stub_hyb_iswa_get_cache_size(const llama_memory_hybrid_iswa &, uint64_t, uint32_t) {
    aidaptiv_not_loaded("hyb_iswa_get_cache_size");
}
void stub_hyb_iswa_kv_cache_read(llama_memory_hybrid_iswa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("hyb_iswa_kv_cache_read");
}
void stub_hyb_iswa_kv_cache_write(llama_memory_hybrid_iswa &, void *, const size_t &, const uint32_t &, const llama_seq_id &, const llama_pos &, const size_t &, bool, uint32_t, const llama_pos *) {
    aidaptiv_not_loaded("hyb_iswa_kv_cache_write");
}
void stub_hyb_iswa_get_cached_positions(const llama_memory_hybrid_iswa &, const llama_seq_id &, const size_t &, bool *, uint32_t) {
    aidaptiv_not_loaded("hyb_iswa_get_cached_positions");
}

void stub_model_setup_moe_offload(llama_model_base &, int, aidaptiv_moe_offload_params *) {
    aidaptiv_not_loaded("model_setup_moe_offload");
}
void stub_model_clear_moe_offload(const llama_model_base &) {
}
bool stub_model_is_moe_offload_enabled(const llama_model_base &) {
    aidaptiv_not_loaded("model_is_moe_offload_enabled");
}
bool stub_model_need_exclude(llama_model_base &, uint32_t) {
    aidaptiv_not_loaded("model_need_exclude");
}
std::unordered_map<std::string, ggml_tensor *> stub_model_distribute_expert_tensor(llama_model_base &, llama_model_base::expert_tensor_params, uint32_t, const std::function<bool(uint32_t)> &) {
    aidaptiv_not_loaded("model_distribute_expert_tensor");
}
void stub_model_offload_expert(llama_model_base &, llama_model_base::expert_tensor_params, uint32_t, const std::function<bool(uint32_t)> &, uint32_t) {
    aidaptiv_not_loaded("model_offload_expert");
}
void stub_model_create_expert_manager(llama_model_base &, std::unordered_map<std::string, ggml_tensor *> &, llama_model_base::expert_tensor_params, uint32_t, const std::function<bool(uint32_t)> &) {
    aidaptiv_not_loaded("model_create_expert_manager");
}

void stub_em_destroy(ExpertManager *) {
    aidaptiv_not_loaded("em_destroy");
}
bool stub_em_preload_experts(ExpertManager *) {
    aidaptiv_not_loaded("em_preload_experts");
}
void stub_em_set_tensor(ExpertManager *, ggml_tensor *, uint32_t) {
    aidaptiv_not_loaded("em_set_tensor");
}
float stub_em_get_prefill_vram_hit_rate(const ExpertManager *) {
    aidaptiv_not_loaded("em_get_prefill_vram_hit_rate");
}
float stub_em_get_prefill_dram_hit_rate(const ExpertManager *) {
    aidaptiv_not_loaded("em_get_prefill_dram_hit_rate");
}
float stub_em_get_decode_vram_hit_rate(const ExpertManager *) {
    aidaptiv_not_loaded("em_get_decode_vram_hit_rate");
}
float stub_em_get_decode_dram_hit_rate(const ExpertManager *) {
    aidaptiv_not_loaded("em_get_decode_dram_hit_rate");
}
void stub_em_reset_hit_rate(ExpertManager *) {
    aidaptiv_not_loaded("em_reset_hit_rate");
}
ggml_tensor * stub_em_schedule_experts(ggml_context *, ggml_tensor *, ExpertManager *, int) {
    aidaptiv_not_loaded("em_schedule_experts");
}

const aidaptiv_dispatch stub_dispatch = {
    stub_kv_cache_get_cache_size,
    stub_kv_cache_read,
    stub_kv_cache_write,
    stub_kv_cache_get_cached_positions,

    stub_mr_get_size,
    stub_mr_get_cache_size,
    stub_mr_kv_cache_read,
    stub_mr_kv_cache_write,
    stub_mr_get_cached_positions,

    stub_iswa_get_size,
    stub_iswa_get_cache_size,
    stub_iswa_kv_cache_read,
    stub_iswa_kv_cache_write,
    stub_iswa_get_cached_positions,

    stub_dsa_get_size,
    stub_dsa_get_cache_size,
    stub_dsa_kv_cache_read,
    stub_dsa_kv_cache_write,
    stub_dsa_get_cached_positions,

    stub_hyb_get_size,
    stub_hyb_get_cache_size,
    stub_hyb_kv_cache_read,
    stub_hyb_kv_cache_write,
    stub_hyb_get_cached_positions,

    stub_hyb_iswa_get_size,
    stub_hyb_iswa_get_cache_size,
    stub_hyb_iswa_kv_cache_read,
    stub_hyb_iswa_kv_cache_write,
    stub_hyb_iswa_get_cached_positions,

    stub_model_setup_moe_offload,
    stub_model_clear_moe_offload,
    stub_model_is_moe_offload_enabled,
    stub_model_need_exclude,
    stub_model_distribute_expert_tensor,
    stub_model_offload_expert,
    stub_model_create_expert_manager,

    stub_em_destroy,
    stub_em_preload_experts,
    stub_em_set_tensor,
    stub_em_get_prefill_vram_hit_rate,
    stub_em_get_prefill_dram_hit_rate,
    stub_em_get_decode_vram_hit_rate,
    stub_em_get_decode_dram_hit_rate,
    stub_em_reset_hit_rate,
    stub_em_schedule_experts,
};

}  // namespace

const aidaptiv_dispatch * g_aidaptiv = &stub_dispatch;

static void abi_log_field(const char * name, size_t aidaptiv_val, size_t host_val) {
    if (aidaptiv_val != host_val) {
        std::fprintf(stderr,
                     "[ABI][FATAL] %s: aidaptiv=%zu host=%zu (delta=%+lld)\n",
                     name, aidaptiv_val, host_val,
                     (long long) aidaptiv_val - (long long) host_val);
    }
}

void llama_register_aidaptiv_dispatch(const aidaptiv_dispatch *        disp,
                                       const aidaptiv_abi_fingerprint * aidaptiv_fp) {
    aidaptiv_abi_fingerprint host_fp;
    fill_aidaptiv_abi_fingerprint(host_fp);

    if (aidaptiv_fp == nullptr) {
        std::fprintf(stderr,
                     "[ABI][FATAL] aidaptiv_core did not provide an ABI fingerprint; "
                     "rebuild aidaptiv.dll against a matching aidaptiv-dispatch.h\n");
        std::abort();
    }

    if (aidaptiv_fp->magic != host_fp.magic) {
        std::fprintf(stderr,
                     "[ABI][FATAL] fingerprint magic mismatch: aidaptiv=0x%08X host=0x%08X "
                     "(aidaptiv.dll appears to be built against a foreign aidaptiv-dispatch.h)\n",
                     aidaptiv_fp->magic, host_fp.magic);
        std::abort();
    }

    if (aidaptiv_fp->version != host_fp.version) {
        std::fprintf(stderr,
                     "[ABI][FATAL] fingerprint version mismatch: aidaptiv=%u host=%u "
                     "(one side is built against an older/newer aidaptiv-dispatch.h)\n",
                     aidaptiv_fp->version, host_fp.version);
        std::abort();
    }

#define AIDP_CHECK(field) abi_log_field(#field, aidaptiv_fp->field, host_fp.field)
    AIDP_CHECK(sizeof_aidaptiv_dispatch);
    AIDP_CHECK(sizeof_aidaptiv_moe_offload_params);
    AIDP_CHECK(sizeof_llama_model_params);
    AIDP_CHECK(sizeof_llama_model_base);
    AIDP_CHECK(sizeof_llama_kv_cache);
    AIDP_CHECK(sizeof_llama_kv_cache_iswa);
    AIDP_CHECK(sizeof_llama_kv_cache_dsa);
    AIDP_CHECK(sizeof_llama_memory_recurrent);
    AIDP_CHECK(sizeof_llama_memory_hybrid);
    AIDP_CHECK(sizeof_llama_memory_hybrid_iswa);
    AIDP_CHECK(sizeof_llama_hparams);
    AIDP_CHECK(offsetof_llama_model_params__offload_folder);
    AIDP_CHECK(offsetof_llama_model_params__temp_uuid);
    AIDP_CHECK(offsetof_llama_model_params__vram_experts_cached_gb);
#undef AIDP_CHECK

    if (std::memcmp(aidaptiv_fp, &host_fp, sizeof(host_fp)) != 0) {
        std::fprintf(stderr,
                     "[ABI][FATAL] aidaptiv_core ABI mismatch detected; aborting before "
                     "memory corruption can occur. See per-field diff above and rebuild both "
                     "aidaptiv.dll and llama.dll from matching headers.\n");
        std::abort();
    }

    g_aidaptiv = disp != nullptr ? disp : &stub_dispatch;
}
