#include "aidaptiv.h"

#include "log.h"

#ifdef LLAMA_USE_AIDAPTIV
#include "aidaptiv.hpp"
#endif

#include <stdexcept>
#include <utility>

static std::vector<common_adapter_lora_info> common_aidaptiv_active_loras(const std::vector<common_adapter_lora_info> & lora_adapters) {
    std::vector<common_adapter_lora_info> res;
    for (const auto & la : lora_adapters) {
        if (la.ptr != nullptr && la.scale != 0.0f) {
            res.push_back(la);
        }
    }
    return res;
}

#ifdef LLAMA_USE_AIDAPTIV

static std::vector<aidaptiv::lora_info> common_aidaptiv_lora_info(const std::vector<common_adapter_lora_info> & lora_adapters) {
    std::vector<aidaptiv::lora_info> res;
    const auto active_loras = common_aidaptiv_active_loras(lora_adapters);
    res.reserve(active_loras.size());
    for (const auto & la : active_loras) {
        res.push_back(aidaptiv::lora_info{
            la.path,
            la.scale,
            la.task_name,
            la.prompt_prefix,
        });
    }
    return res;
}

static std::vector<aidaptiv::mtmd_chunk_info> common_aidaptiv_mtmd_info(const std::vector<common_aidaptiv_mtmd_chunk_info> & mtmd_info) {
    std::vector<aidaptiv::mtmd_chunk_info> res;
    res.reserve(mtmd_info.size());
    for (const auto & cur : mtmd_info) {
        res.push_back(aidaptiv::mtmd_chunk_info{
            cur.hash,
            cur.token_cnt,
        });
    }
    return res;
}

struct common_aidaptiv::impl {
    bool active = false;
    std::string offload_path;
    std::unique_ptr<aidaptiv::Aidaptiv> runtime;
};

common_aidaptiv::common_aidaptiv(const common_params & params, std::string & resolved_debug_log_path) :
    pimpl(new impl{}) {
    pimpl->active = true;

    aidaptiv::setup_params setup;
    setup.ssd_kv_offload_gb      = params.phison_ssd_kv_offload_gb;
    setup.dram_kv_offload_gb     = params.phison_dram_kv_offload_gb;
    setup.kv_cache_resume_policy = static_cast<uint32_t>(params.phison_kv_cache_resume_policy);
    setup.flash_attn             = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED;
    setup.model_path             = params.model.path;
    setup.mmproj_model_path      = params.mmproj.path;

    const std::string offload_path;
    pimpl->runtime.reset(new aidaptiv::Aidaptiv(offload_path, resolved_debug_log_path, setup));
    pimpl->offload_path = pimpl->runtime->offload_path();

    LOG_INF("%s: aiDAPTIV enabled, version: %s, offload path: %s\n",
            __func__, pimpl->runtime->version().c_str(), pimpl->offload_path.c_str());
}

common_aidaptiv::~common_aidaptiv() = default;

bool common_aidaptiv::enabled() const {
    return pimpl->active;
}

uint64_t common_aidaptiv::generate_uuid() {
    if (!pimpl->runtime) {
        return 0;
    }
    return pimpl->runtime->generate_uuid();
}

const std::string & common_aidaptiv::offload_path() const {
    return pimpl->offload_path;
}

void common_aidaptiv::init(llama_context * ctx, llama_model * model, const std::vector<llama_adapter_lora_ptr> & lora_init, const std::vector<common_adapter_lora_info> & lora_adapters) {
    if (!pimpl->runtime) {
        return;
    }

    std::vector<llama_adapter_lora *> lora_ptrs;
    lora_ptrs.reserve(lora_init.size());
    for (const auto & la : lora_init) {
        lora_ptrs.push_back(la.get());
    }

    pimpl->runtime->init(ctx, model, lora_ptrs, common_aidaptiv_lora_info(lora_adapters));
}

common_aidaptiv_restore_stats common_aidaptiv::restore_kv_cache(
        const llama_tokens & prompt_tokens,
        const std::vector<common_aidaptiv_mtmd_chunk_info> & mtmd_info,
        const std::vector<common_adapter_lora_info> & lora_adapters,
        uint32_t slot_id,
        uint32_t hit_in_device) {
    if (!pimpl->runtime) {
        return {};
    }

    auto stats = pimpl->runtime->restore_kv_cache(
            prompt_tokens,
            common_aidaptiv_mtmd_info(mtmd_info),
            common_aidaptiv_lora_info(lora_adapters),
            slot_id,
            hit_in_device);

    return { stats.dram_reuse_token_cnt, stats.ssd_reuse_token_cnt };
}

void common_aidaptiv::save_kv_cache(
        const llama_tokens & tokens,
        const std::vector<common_aidaptiv_mtmd_chunk_info> & mtmd_info,
        const std::vector<common_adapter_lora_info> & lora_adapters,
        uint32_t slot_id) {
    if (!pimpl->runtime || tokens.empty()) {
        return;
    }

    pimpl->runtime->save_kv_cache(
            tokens,
            common_aidaptiv_mtmd_info(mtmd_info),
            common_aidaptiv_lora_info(lora_adapters),
            slot_id);
}

void common_aidaptiv::flush_kv_cache() {
    if (pimpl->runtime) {
        pimpl->runtime->flush_kv_cache();
    }
}

void common_aidaptiv::remove_temp_caches() {
    if (pimpl->runtime) {
        pimpl->runtime->remove_temp_caches();
    }
}

#else

struct common_aidaptiv::impl {
    bool active = false;
    std::string offload_path;
};

common_aidaptiv::common_aidaptiv(const common_params & params, std::string & resolved_debug_log_path) :
    pimpl(new impl{}) {
    GGML_UNUSED(params);
    GGML_UNUSED(resolved_debug_log_path);
}

common_aidaptiv::~common_aidaptiv() = default;

bool common_aidaptiv::enabled() const {
    return false;
}

uint64_t common_aidaptiv::generate_uuid() {
    return 0;
}

const std::string & common_aidaptiv::offload_path() const {
    return pimpl->offload_path;
}

void common_aidaptiv::init(llama_context *, llama_model *, const std::vector<llama_adapter_lora_ptr> &, const std::vector<common_adapter_lora_info> &) {
}

common_aidaptiv_restore_stats common_aidaptiv::restore_kv_cache(
        const llama_tokens &,
        const std::vector<common_aidaptiv_mtmd_chunk_info> &,
        const std::vector<common_adapter_lora_info> &,
        uint32_t,
        uint32_t) {
    return {};
}

void common_aidaptiv::save_kv_cache(
        const llama_tokens &,
        const std::vector<common_aidaptiv_mtmd_chunk_info> &,
        const std::vector<common_adapter_lora_info> &,
        uint32_t) {
}

void common_aidaptiv::flush_kv_cache() {
}

void common_aidaptiv::remove_temp_caches() {
}

#endif

