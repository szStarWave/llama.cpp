#pragma once

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <cstddef>
#include <cstdint>

class ExpertManager;

struct aidaptiv_abi_fingerprint {
    uint32_t magic;
    uint32_t version;

    // sizeof of structs that cross the dispatch boundary
    size_t   sizeof_aidaptiv_dispatch;
    size_t   sizeof_aidaptiv_moe_offload_params;
    size_t   sizeof_llama_model_params;
    size_t   sizeof_llama_model_base;
    size_t   sizeof_llama_kv_cache;
    size_t   sizeof_llama_kv_cache_iswa;
    size_t   sizeof_llama_kv_cache_dsa;
    size_t   sizeof_llama_memory_recurrent;
    size_t   sizeof_llama_memory_hybrid;
    size_t   sizeof_llama_memory_hybrid_iswa;
    size_t   sizeof_llama_hparams;

    // offsetof of fields aidaptiv-core dereferences (most common drift source)
    size_t   offsetof_llama_model_params__offload_folder;
    size_t   offsetof_llama_model_params__temp_uuid;
    size_t   offsetof_llama_model_params__vram_experts_cached_gb;
};

inline void fill_aidaptiv_abi_fingerprint(aidaptiv_abi_fingerprint & f);

struct aidaptiv_dispatch {
    // ---- llama_kv_cache ----
    size_t (*kv_cache_get_cache_size )(const llama_kv_cache & self, uint64_t node_size, uint32_t mask);
    void   (*kv_cache_read           )(llama_kv_cache & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void   (*kv_cache_write          )(llama_kv_cache & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void   (*kv_cache_get_cached_positions)(const llama_kv_cache & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_memory_recurrent ----
    uint32_t (*mr_get_size          )(const llama_memory_recurrent & self);
    size_t   (*mr_get_cache_size    )(const llama_memory_recurrent & self, uint64_t node_size, uint32_t mask);
    void     (*mr_kv_cache_read     )(const llama_memory_recurrent & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*mr_kv_cache_write    )(llama_memory_recurrent & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*mr_get_cached_positions)(const llama_memory_recurrent & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_kv_cache_iswa ----
    uint32_t (*iswa_get_size          )(const llama_kv_cache_iswa & self);
    size_t   (*iswa_get_cache_size    )(const llama_kv_cache_iswa & self, uint64_t node_size, uint32_t mask);
    void     (*iswa_kv_cache_read     )(llama_kv_cache_iswa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*iswa_kv_cache_write    )(llama_kv_cache_iswa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*iswa_get_cached_positions)(const llama_kv_cache_iswa & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_kv_cache_dsa ----
    uint32_t (*dsa_get_size          )(const llama_kv_cache_dsa & self);
    size_t   (*dsa_get_cache_size    )(const llama_kv_cache_dsa & self, uint64_t node_size, uint32_t mask);
    void     (*dsa_kv_cache_read     )(llama_kv_cache_dsa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*dsa_kv_cache_write    )(llama_kv_cache_dsa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*dsa_get_cached_positions)(const llama_kv_cache_dsa & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_memory_hybrid ----
    uint32_t (*hyb_get_size          )(const llama_memory_hybrid & self);
    size_t   (*hyb_get_cache_size    )(const llama_memory_hybrid & self, uint64_t node_size, uint32_t mask);
    void     (*hyb_kv_cache_read     )(llama_memory_hybrid & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*hyb_kv_cache_write    )(llama_memory_hybrid & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*hyb_get_cached_positions)(const llama_memory_hybrid & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_memory_hybrid_iswa ----
    uint32_t (*hyb_iswa_get_size          )(const llama_memory_hybrid_iswa & self);
    size_t   (*hyb_iswa_get_cache_size    )(const llama_memory_hybrid_iswa & self, uint64_t node_size, uint32_t mask);
    void     (*hyb_iswa_kv_cache_read     )(llama_memory_hybrid_iswa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*hyb_iswa_kv_cache_write    )(llama_memory_hybrid_iswa & self, void * ptr, const size_t & node_stride, const uint32_t & node_size, const llama_seq_id & seq_id, const llama_pos & start_pos, const size_t & count, bool is_last_node, uint32_t mask, const llama_pos * mrope_pos);
    void     (*hyb_iswa_get_cached_positions)(const llama_memory_hybrid_iswa & self, const llama_seq_id & seq_id, const size_t & count, bool * cached, uint32_t mask);

    // ---- llama_model_base ----
    void (*model_setup_moe_offload       )(llama_model_base & self, int n_gpu_layers, aidaptiv_moe_offload_params * moe);
    void (*model_clear_moe_offload       )(const llama_model_base & self);
    bool (*model_is_moe_offload_enabled  )(const llama_model_base & self);
    bool (*model_need_exclude            )(llama_model_base & self, uint32_t il);
    std::unordered_map<std::string, ggml_tensor *> 
         (*model_distribute_expert_tensor)(llama_model_base & self, llama_model_base::expert_tensor_params t_list, uint32_t n_layers, const std::function<bool(uint32_t)> & is_moe_layer);
    void (*model_offload_expert          )(llama_model_base & self, llama_model_base::expert_tensor_params t_list, uint32_t n_layers, const std::function<bool(uint32_t)> & is_moe_layer, uint32_t n_expert);
    void (*model_create_expert_manager   )(llama_model_base & self, std::unordered_map<std::string, ggml_tensor *> & mapping_table, llama_model_base::expert_tensor_params t_list, uint32_t n_layers, const std::function<bool(uint32_t)> & is_moe_layer);

    // ---- ExpertManager ----
    void  (*em_destroy                   )(ExpertManager * mgr);
    bool  (*em_preload_experts           )(ExpertManager * mgr);
    void  (*em_set_tensor                )(ExpertManager * mgr, ggml_tensor * tensor, uint32_t il);
    float (*em_get_single_token_vram_hit_rate)(const ExpertManager * mgr);
    float (*em_get_single_token_dram_hit_rate)(const ExpertManager * mgr);
    float (*em_get_multi_token_vram_hit_rate )(const ExpertManager * mgr);
    float (*em_get_multi_token_dram_hit_rate )(const ExpertManager * mgr);
    float (*em_get_vram_hit_rate             )(const ExpertManager * mgr);
    float (*em_get_dram_hit_rate             )(const ExpertManager * mgr);
    void  (*em_reset_hit_rate                )(ExpertManager * mgr);
    ggml_tensor * (*em_schedule_experts  )(ggml_context * ctx, ggml_tensor * route_output, ExpertManager * mgr, int il);
};

extern const aidaptiv_dispatch * g_aidaptiv;

LLAMA_API void llama_register_aidaptiv_dispatch(const aidaptiv_dispatch *        disp,
                                                 const aidaptiv_abi_fingerprint * aidaptiv_fp);

inline void fill_aidaptiv_abi_fingerprint(aidaptiv_abi_fingerprint & f) {
    f.magic   = 0x41494450u;
    f.version = 2u;

    f.sizeof_aidaptiv_dispatch              = sizeof(aidaptiv_dispatch);
    f.sizeof_aidaptiv_moe_offload_params    = sizeof(aidaptiv_moe_offload_params);
    f.sizeof_llama_model_params             = sizeof(llama_model_params);
    f.sizeof_llama_model_base               = sizeof(llama_model_base);
    f.sizeof_llama_kv_cache                 = sizeof(llama_kv_cache);
    f.sizeof_llama_kv_cache_iswa            = sizeof(llama_kv_cache_iswa);
    f.sizeof_llama_kv_cache_dsa             = sizeof(llama_kv_cache_dsa);
    f.sizeof_llama_memory_recurrent         = sizeof(llama_memory_recurrent);
    f.sizeof_llama_memory_hybrid            = sizeof(llama_memory_hybrid);
    f.sizeof_llama_memory_hybrid_iswa       = sizeof(llama_memory_hybrid_iswa);
    f.sizeof_llama_hparams                  = sizeof(llama_hparams);

    f.offsetof_llama_model_params__offload_folder         = offsetof(llama_model_params, offload_folder);
    f.offsetof_llama_model_params__temp_uuid              = offsetof(llama_model_params, temp_uuid);
    f.offsetof_llama_model_params__vram_experts_cached_gb = offsetof(llama_model_params, vram_experts_cached_gb);
}
