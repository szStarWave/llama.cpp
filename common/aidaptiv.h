#pragma once

#include "common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct common_aidaptiv_mtmd_chunk_info {
    uint64_t hash      = 0;
    size_t   token_cnt = 0;
};

struct common_aidaptiv_restore_stats {
    uint32_t dram_reuse_token_cnt = 0;
    uint32_t ssd_reuse_token_cnt  = 0;
};

class common_aidaptiv {
public:
    common_aidaptiv(const common_params & params, std::string & resolved_debug_log_path);
    ~common_aidaptiv();

    bool enabled() const;
    uint64_t generate_uuid();
    const std::string & offload_path() const;

    void init(llama_context * ctx, llama_model * model, const std::vector<llama_adapter_lora_ptr> & lora_init, const std::vector<common_adapter_lora_info> & lora_adapters);

    common_aidaptiv_restore_stats restore_kv_cache(
            const llama_tokens & prompt_tokens,
            const std::vector<common_aidaptiv_mtmd_chunk_info> & mtmd_info,
            const std::vector<common_adapter_lora_info> & lora_adapters,
            uint32_t slot_id,
            uint32_t hit_in_device);

    void save_kv_cache(
            const llama_tokens & tokens,
            const std::vector<common_aidaptiv_mtmd_chunk_info> & mtmd_info,
            const std::vector<common_adapter_lora_info> & lora_adapters,
            uint32_t slot_id);

    void flush_kv_cache();
    void remove_temp_caches();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

