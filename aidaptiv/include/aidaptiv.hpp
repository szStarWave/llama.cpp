#ifndef _AIDAPTIV_HPP
#define _AIDAPTIV_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef AIDAPTIV_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef AIDAPTIV_BUILD
#            define AIDAPTIV_API __declspec(dllexport)
#        else
#            define AIDAPTIV_API __declspec(dllimport)
#        endif
#    else
#        define AIDAPTIV_API __attribute__((visibility("default")))
#    endif
#else
#    define AIDAPTIV_API
#endif

struct llama_context;
struct llama_model;
struct llama_adapter_lora;

namespace aidaptiv {

using token_id  = int32_t;
using token_seq = std::vector<token_id>;


struct mtmd_chunk_info {
    uint64_t    img_chunk_id      = 0;
    std::size_t token_cnt = 0;
    std::vector<int32_t> mrope_pos;
    uint32_t             mrope_n_pos_advance = 0;
};
using mtmd_seq_info = std::vector<mtmd_chunk_info>;

struct lora_info {
    std::string path;
    float       scale = 0.0f;
    std::string task_name;
    std::string prompt_prefix;
};

// KV / resume settings for Aidaptiv(offload_path, debug_log_path, params).
// MoE expert offload is configured via llama_model_params (resolved in llama-model).
struct setup_params {
    int32_t  ssd_kv_offload_gb      = 0;
    int32_t  dram_kv_offload_gb     = 0;
    uint32_t kv_cache_resume_policy = 0;
    bool dry_run = false;

    // Filled in by init() after llama_context exists
    uint32_t total_ctx_size = 0;
    bool     flash_attn     = false;

    std::string model_path;
    std::string mmproj_model_path;
};

struct kv_restore_stats {
    uint32_t dram_reuse_token_cnt = 0;
    uint32_t ssd_reuse_token_cnt  = 0;
};

class AIDAPTIV_API Aidaptiv {
  public:
    /**
     * @brief Construct an Aidaptiv runtime.
     *
     * @param offload_path    Root directory used to store KV-cache data.
     *                        Must exist and be writable.
     * @param debug_log_path  Path used for the debug log
     *                        (in/out: may be normalised internally).
     * @param params          KV / resume settings. See @ref setup_params.
     *
     * @throw std::runtime_error on initialisation failure.
     */
    Aidaptiv(const std::string & offload_path,
             std::string & debug_log_path,
             const setup_params & params = {});

    ~Aidaptiv();
    Aidaptiv(Aidaptiv &&) noexcept             = default;
    Aidaptiv & operator=(Aidaptiv &&) noexcept = default;
    Aidaptiv(const Aidaptiv &)                 = delete;
    Aidaptiv & operator=(const Aidaptiv &)     = delete;

    /**
     * @brief Get the offload root directory.
     * @return Reference to the path string passed to the constructor.
     */
    const std::string & offload_path() const;

    /**
     * @brief KV-cache setup that requires a live @c llama_context.
     *
     * Must be called after @c llama_init_from_model() and before the first
     * inference call. If @p ctx or @p model is null, the call is a no-op.
     *
     * @param ctx           Active @c llama_context (must be non-null).
     * @param model         Loaded @c llama_model (must be non-null).
     * @param lora_init     LoRA adapters already loaded into llama.cpp.
     * @param lora_adapters Logical LoRA descriptors used as part of the
     *                      KV cache key.
     */
    void init(llama_context *                           ctx,
              llama_model *                             model,
              const std::vector<llama_adapter_lora *> & lora_init     = {},
              const std::vector<lora_info> &            lora_adapters = {});

    /** @brief Remove every unlocked cache under the offload root. */
    void                  remove_temp_caches();

    /** @brief Remove only the expert caches this run published/adopted, skipping
     *         any still held by another live process. Keeps other runs' caches for
     *         resume. */
    void                  remove_owned_temp_caches();

    /** @brief Library/build version string. */
    std::string           version();

    /**
     * @brief Try to restore KV-cache for a prompt prefix.
     *
     * Looks up the longest matching prefix of @p prompt_tokens (combined
     * with @p mtmd_info and @p lora_adapters) and refills the corresponding
     * cells of the model's KV cache for slot @p slot_id.
     *
     * @param prompt_tokens  Full prompt token sequence to restore against.
     * @param mtmd_info      Multimodal chunks interleaved in the prompt;
     *                       pass @c {} for pure-text prompts.
     * @param lora_adapters  LoRA descriptors that participate in the key.
     * @param slot_id        Sequence id within the @c llama_context KV cache.
     * @param hit_in_device  Number of leading prompt tokens whose KV caches are
     *                       already in the GPU (device). Use @c 0 if the device
     *                       has no matching prefix.
     *
     * @return Reuse counts per tier. Caller should skip
     *         @c dram_reuse_token_cnt + @c ssd_reuse_token_cnt leading
     *         tokens when forming the next batch.
     */
    kv_restore_stats restore_kv_cache(const token_seq &              prompt_tokens,
                                      const mtmd_seq_info &          mtmd_info,
                                      const std::vector<lora_info> & lora_adapters,
                                      uint32_t                       slot_id       = 0,
                                      uint32_t                       hit_in_device = 0);

    /**
     * @brief Save the KV-cache of @p tokens for future reuse.
     *
     * @param tokens         Token sequence whose KV should be persisted.
     * @param mtmd_info      Multimodal chunks interleaved in @p tokens.
     * @param lora_adapters  LoRA descriptors that participate in the key.
     * @param slot_id        Sequence id within the @c llama_context KV cache.
     * @param subfolder      If empty, save into the default KV cache;
     *                       otherwise save into that named subfolder under
     *                       @ref offload_path().
     */
    void save_kv_cache(const token_seq &              tokens,
                       const mtmd_seq_info &          mtmd_info,
                       const std::vector<lora_info> & lora_adapters,
                       uint32_t                       seq_id   = 0,
                       const std::string &            subfolder = "");

    /** @brief Flush any pending KV-cache writes to disk. */
    void flush_kv_cache();

    /**
     * @brief Snapshot of cache subfolders and their lock states.
     * @return Map from subfolder name to a stringified lock state.
     */
    std::unordered_map<std::string, std::string> get_lock_folder_map();

    /**
     * @brief Update lock state for one or more cache subfolders.
     *
     * @param set_lock_map Map of subfolder name -> desired lock state
     *                     (true = lock, false = unlock).
     * @return Diagnostic string (empty on success).
     */
    std::string update_lock_folders(const std::unordered_map<std::string, bool> & set_lock_map);

    /**
     * @brief Free space under the offload root by evicting cached entries.
     *
     * @param expect_size  Target number of bytes to free.
     * @return Number of bytes actually reclaimed.
     */
    std::size_t clean_kv_cache(int64_t expect_size);

  private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace aidaptiv

#endif
