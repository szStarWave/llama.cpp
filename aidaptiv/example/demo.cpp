//////////////////////////////////////////////////////
//
//  This example is modified from llama-simple
//
//////////////////////////////////////////////////////

#include "llama.h"
#include "llama-ext.h"
#include "aidaptiv.hpp"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -o offload_path [-n n_predict]\n"
           "        [-lora adapter.gguf] [-ls scale]\n"
           "        [-vc vram_experts_cached_gb] [-dc dram_experts_cached_gb]\n"
           "        [-sk ssd_kv_offload_gb] [-dk dram_kv_offload_gb]\n"
           "        [-kr kv_cache_resume_policy] [-fa]\n"
           "        [-mtp] [-md draft.gguf] [-dn n_draft]\n"
           "        [prompt]\n", argv[0]);
    printf("\n  MTP (draft-mtp):\n");
    printf("    Qwen3.5:  -mtp  (same GGUF; optional -md for split MTP head)\n");
    printf("    Gemma4:   -mtp -md gemma4-assistant.gguf\n");
    printf("\n");
}

static bool apply_lora(
        llama_context * ctx,
        const std::vector<llama_adapter_lora *> & lora_init,
        const std::vector<aidaptiv::lora_info> & lora_adapters) {
    if (lora_init.empty()) {
        return true;
    }
    std::vector<float> scales;
    scales.reserve(lora_adapters.size());
    for (const auto & la : lora_adapters) {
        scales.push_back(la.scale);
    }
    const int32_t rc = llama_set_adapters_lora(
        ctx,
        const_cast<llama_adapter_lora **>(lora_init.data()),
        lora_init.size(),
        scales.data());
    if (rc != 0) {
        fprintf(stderr, "%s: llama_set_adapters_lora failed (%d)\n", __func__, rc);
        return false;
    }
    return true;
}

static void batch_clear(llama_batch & batch) {
    batch.n_tokens = 0;
}

static void batch_add(
        llama_batch & batch,
        llama_token id,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits) {
    const int i = batch.n_tokens;
    if (batch.token) {
        batch.token[i] = id;
    }
    batch.pos[i]      = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i]   = logits ? 1 : 0;
    batch.n_tokens++;
}

static void print_token(const llama_vocab * vocab, llama_token id) {
    char buf[128];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return;
    }
    fwrite(buf, 1, n, stdout);
    fflush(stdout);
}

// Hide noisy DEBUG from MTP checkpoint restore (state_read_meta), keep INFO/WARN/ERROR.
static void demo_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
    fputs(text, stderr);
    fflush(stderr);
}

// Minimal single-seq draft-mtp state (from common/speculative.cpp draft_mtp).
struct mtp_state {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    llama_model   * model_dft = nullptr; // non-null only when -md loaded a separate file
    bool            owns_dft_model = false;

    llama_batch     batch_dft = {};
    bool            batch_dft_init = false;
    llama_sampler * smpl_dft  = nullptr;

    int32_t n_embd = 0;
    bool    is_mem_shared = false;
    int32_t n_draft_max = 3;

    std::vector<float> pending_h;
    std::vector<float> verify_h;
    int32_t            verify_h_rows = 0;
};

static void mtp_free(mtp_state & m) {
    if (m.smpl_dft) {
        llama_sampler_free(m.smpl_dft);
        m.smpl_dft = nullptr;
    }
    if (m.batch_dft_init) {
        if (m.batch_dft.token) {
            free(m.batch_dft.token);
            m.batch_dft.token = nullptr;
        }
        llama_batch_free(m.batch_dft);
        m.batch_dft = {};
        m.batch_dft_init = false;
    }
    if (m.ctx_dft) {
        llama_free(m.ctx_dft);
        m.ctx_dft = nullptr;
    }
    if (m.owns_dft_model && m.model_dft) {
        llama_model_free(m.model_dft);
        m.model_dft = nullptr;
    }
}

static bool mtp_init(
        mtp_state & m,
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        const std::string & draft_path,
        const std::string & offload_folder,
        int n_ctx,
        int n_batch,
        int n_draft_max) {
    m.ctx_tgt     = ctx_tgt;
    m.n_draft_max = n_draft_max;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = n_ctx;
    cparams.n_batch       = n_batch;
    cparams.n_rs_seq      = 0;
    cparams.no_perf       = false;
    cparams.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
    cparams.ctx_other     = ctx_tgt;

    if (!draft_path.empty()) {
        llama_model_params mparams = llama_model_default_params();
        mparams.offload_folder         = offload_folder.c_str();
        mparams.vram_experts_cached_gb = 0;
        mparams.dram_experts_cached_gb = 0;

        m.model_dft = llama_model_load_from_file(draft_path.c_str(), mparams);
        if (!m.model_dft) {
            fprintf(stderr, "%s: failed to load draft model '%s'\n", __func__, draft_path.c_str());
            return false;
        }
        m.owns_dft_model = true;

        m.ctx_dft = llama_init_from_model(m.model_dft, cparams);
    } else {
        m.model_dft = model_tgt;
        m.ctx_dft = llama_init_from_model(model_tgt, cparams);
    }

    if (!m.ctx_dft) {
        fprintf(stderr, "%s: failed to create MTP draft context\n", __func__);
        return false;
    }

    m.n_embd = llama_model_n_embd_out(llama_get_model(m.ctx_dft));
    if (m.n_embd != llama_model_n_embd(model_tgt)) {
        fprintf(stderr, "%s: MTP embd width mismatch (dft=%d tgt=%d)\n",
                __func__, m.n_embd, llama_model_n_embd(model_tgt));
        return false;
    }

    const int32_t n_b = (int32_t) llama_n_batch(m.ctx_dft);
    m.batch_dft = llama_batch_init(n_b, m.n_embd, 1);
    m.batch_dft.token = (llama_token *) malloc(sizeof(llama_token) * (size_t) n_b);
    if (!m.batch_dft.token) {
        fprintf(stderr, "%s: failed to allocate draft batch tokens\n", __func__);
        return false;
    }
    m.batch_dft_init = true;

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    m.smpl_dft = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(m.smpl_dft, llama_sampler_init_greedy());

    llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
    llama_set_embeddings_nextn(m.ctx_dft, true, /*masked*/ true);

    // Same check as common/speculative.cpp: only Gemma4 assistant keeps ctx_other.
    m.is_mem_shared = llama_get_ctx_other(m.ctx_dft) == ctx_tgt;
    m.pending_h.assign((size_t) m.n_embd, 0.0f);
    m.verify_h.clear();
    m.verify_h_rows = 0;

    printf("\nMTP: draft context ready (shared_mem=%d, n_draft_max=%d, n_embd=%d)\n",
           (int) m.is_mem_shared, m.n_draft_max, m.n_embd);
    return true;
}

// Mirror target batch into draft KV using shifted h_nextn (skipped when mem shared).
static bool mtp_process(mtp_state & m, const llama_batch & batch_in) {
    if (batch_in.n_tokens <= 0 || batch_in.token == nullptr) {
        return true;
    }

    const int32_t n_tokens = batch_in.n_tokens;
    const size_t  row_bytes = (size_t) m.n_embd * sizeof(float);

    if (!m.is_mem_shared) {
        batch_clear(m.batch_dft);
        for (int k = 0; k < n_tokens; ++k) {
            batch_add(m.batch_dft, batch_in.token[k], batch_in.pos[k], 0, false);
        }

        const float * h_tgt = llama_get_embeddings_nextn(m.ctx_tgt);
        if (!h_tgt) {
            fprintf(stderr, "%s: missing target nextn embeddings\n", __func__);
            return false;
        }
        if (n_tokens > 1) {
            memcpy(m.batch_dft.embd + (size_t) 1 * m.n_embd, h_tgt, row_bytes * (size_t) (n_tokens - 1));
        }
        memcpy(m.batch_dft.embd, m.pending_h.data(), row_bytes);

        if (llama_decode(m.ctx_dft, m.batch_dft) != 0) {
            fprintf(stderr, "%s: draft catch-up decode failed\n", __func__);
            return false;
        }
    }

    m.verify_h_rows = n_tokens;
    m.verify_h.resize((size_t) n_tokens * (size_t) m.n_embd);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * h = llama_get_embeddings_nextn_ith(m.ctx_tgt, i);
        if (!h) {
            fprintf(stderr, "%s: missing nextn embd at %d\n", __func__, i);
            return false;
        }
        memcpy(m.verify_h.data() + (size_t) i * m.n_embd, h, row_bytes);
    }
    memcpy(m.pending_h.data(), m.verify_h.data() + (size_t) (n_tokens - 1) * m.n_embd, row_bytes);
    return true;
}

static std::vector<llama_token> mtp_draft(mtp_state & m, llama_token id_last, llama_pos n_past) {
    std::vector<llama_token> result;
    const size_t row_bytes = (size_t) m.n_embd * sizeof(float);

    batch_clear(m.batch_dft);
    batch_add(m.batch_dft, id_last, n_past, 0, true);
    memcpy(m.batch_dft.embd, m.pending_h.data(), row_bytes);

    if (llama_decode(m.ctx_dft, m.batch_dft) != 0) {
        fprintf(stderr, "%s: draft decode failed\n", __func__);
        return result;
    }

    int i = 0;
    while ((int) result.size() < m.n_draft_max) {
        const llama_token id = llama_sampler_sample(m.smpl_dft, m.ctx_dft, -1);
        llama_sampler_accept(m.smpl_dft, id);
        result.push_back(id);

        if ((int) result.size() >= m.n_draft_max) {
            break;
        }

        const float * h_row = llama_get_embeddings_nextn_ith(m.ctx_dft, 0);
        if (!h_row) {
            break;
        }

        batch_clear(m.batch_dft);
        const llama_pos pos = m.is_mem_shared ? n_past : (n_past + i + 1);
        batch_add(m.batch_dft, id, pos, 0, true);
        memcpy(m.batch_dft.embd, h_row, row_bytes);

        if (llama_decode(m.ctx_dft, m.batch_dft) != 0) {
            fprintf(stderr, "%s: draft step %d failed\n", __func__, i);
            break;
        }
        ++i;
    }

    return result;
}

static std::vector<llama_token> sample_and_accept_n(
        llama_sampler * smpl,
        llama_context * ctx,
        const std::vector<llama_token> & draft) {
    std::vector<llama_token> result;
    result.reserve(draft.size() + 1);

    size_t i = 0;
    for (; i < draft.size(); ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, (int32_t) i);
        llama_sampler_accept(smpl, id);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        const llama_token id = llama_sampler_sample(smpl, ctx, (int32_t) i);
        llama_sampler_accept(smpl, id);
        result.push_back(id);
    }
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    llama_log_set(demo_log_callback, nullptr);

    // path to the model gguf file
    std::string model_path;
    std::string prompt = "Hello my name is";
    int n_predict = 32;
    std::string offload_path = "";
    int vram_experts_cached_gb = 0;
    int dram_experts_cached_gb = 0;

    int  ssd_kv_offload_gb      = 0;
    int  dram_kv_offload_gb     = 0;
    int  kv_cache_resume_policy = 1;
    bool flash_attn             = true;
    std::string lora_path;
    float       lora_scale      = 1.0f;

    bool        enable_mtp = false;
    std::string draft_path;
    int         n_draft_max = 3;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-o") == 0) {
                if (i + 1 < argc) {
                    offload_path = std::string(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-vc") == 0) {
                if (i + 1 < argc) {
                    try {
                        vram_experts_cached_gb = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-dc") == 0) {
                if (i + 1 < argc) {
                    try {
                        dram_experts_cached_gb = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-sk") == 0) {
                if (i + 1 < argc) {
                    try {
                        ssd_kv_offload_gb = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-dk") == 0) {
                if (i + 1 < argc) {
                    try {
                        dram_kv_offload_gb = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-kr") == 0) {
                if (i + 1 < argc) {
                    try {
                        kv_cache_resume_policy = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-fa") == 0) {
                flash_attn = true;
            } else if (strcmp(argv[i], "-lora") == 0) {
                if (i + 1 < argc) {
                    lora_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ls") == 0) {
                if (i + 1 < argc) {
                    try {
                        lora_scale = std::stof(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-mtp") == 0) {
                enable_mtp = true;
            } else if (strcmp(argv[i], "-md") == 0) {
                if (i + 1 < argc) {
                    draft_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-dn") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_draft_max = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                    if (n_draft_max < 1) {
                        n_draft_max = 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                break;
            }
        }
        if (model_path.empty() || offload_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (!enable_mtp && !draft_path.empty()) {
            fprintf(stderr, "%s: -md requires -mtp\n", __func__);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    ggml_backend_load_all();

    aidaptiv::setup_params sp;
    sp.ssd_kv_offload_gb      = ssd_kv_offload_gb;
    sp.dram_kv_offload_gb     = dram_kv_offload_gb;
    sp.kv_cache_resume_policy = static_cast<uint32_t>(kv_cache_resume_policy);
    sp.flash_attn             = flash_attn;
    sp.model_path             = model_path;

    aidaptiv::Aidaptiv adptv(offload_path, offload_path, sp);
    printf("Aidaptiv Version: %s\n", adptv.version().c_str());

    // make sure to get right offload path from aidaptiv in hide ssd case.
    const std::string & resolved_offload = adptv.offload_path();
    // initialize the model

    // Moe offload setting
    llama_model_params model_params = llama_model_default_params();
    model_params.offload_folder         = resolved_offload.c_str();
    model_params.vram_experts_cached_gb = vram_experts_cached_gb;
    model_params.dram_experts_cached_gb = dram_experts_cached_gb;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    std::vector<llama_adapter_lora *> lora_init;
    std::vector<aidaptiv::lora_info>  lora_adapters;
    llama_adapter_lora *              lora = nullptr;

    if (!lora_path.empty()) {
        lora = llama_adapter_lora_init(model, lora_path.c_str());
        if (lora == nullptr) {
            fprintf(stderr, "%s: error: unable to load LoRA '%s'\n", __func__, lora_path.c_str());
            llama_model_free(model);
            return 1;
        }
        lora_init.push_back(lora);
        lora_adapters.push_back({ lora_path, lora_scale, "", "" });
        printf("LoRA: %s (scale=%.3f)\n", lora_path.c_str(), lora_scale);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    const int n_batch_need = enable_mtp
        ? std::max(n_prompt, 1 + n_draft_max)
        : n_prompt;

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx   = n_prompt + n_predict + (enable_mtp ? n_draft_max : 0);
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_batch_need;
    ctx_params.n_rs_seq = enable_mtp ? (uint32_t) n_draft_max : 0u;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    if (!apply_lora(ctx, lora_init, lora_adapters)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    adptv.init(ctx, model, lora_init, lora_adapters);

    // initialize MTP draft context when -mtp is set
    mtp_state mtp;
    std::unique_ptr<aidaptiv::Aidaptiv> adptv_dft;
    if (enable_mtp) {
        if (!mtp_init(mtp, ctx, model, draft_path, resolved_offload,
                      (int) ctx_params.n_ctx, n_batch_need, n_draft_max)) {
            mtp_free(mtp);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (!mtp.is_mem_shared) {
            aidaptiv::setup_params sp_dft = sp;
            sp_dft.model_path = draft_path.empty() ? model_path : draft_path;
            sp_dft.flash_attn = llama_flash_attn(mtp.ctx_dft);

            std::string dft_debug_log = resolved_offload;
            try {
                adptv_dft = std::make_unique<aidaptiv::Aidaptiv>(resolved_offload, dft_debug_log, sp_dft);
                adptv_dft->init(mtp.ctx_dft, mtp.model_dft);
            } catch (const std::exception & e) {
                fprintf(stderr, "%s: MTP KV cache disabled: %s\n", __func__, e.what());
                adptv_dft.reset();
            }
        } else {
            printf("MTP KV cache: draft shares the target memory, restored with the target\n");
        }
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        print_token(vocab, id);
    }

    // always decode the last token to get fresh logits for sampling.
    std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);

    // The draft KV must cover the same prefix as the target, otherwise the draft
    // attends over a hole and every prediction gets rejected.
    uint32_t dft_reuse_token_cnt = 0;
    if (adptv_dft) {
        auto dft_stats = adptv_dft->restore_kv_cache(lookup_tokens, {}, {}, 0, 0);
        dft_reuse_token_cnt = dft_stats.dram_reuse_token_cnt + dft_stats.ssd_reuse_token_cnt;
        printf("\nMTP dram reuse token : %d \nMTP cache reuse token : %d\n",
               dft_stats.dram_reuse_token_cnt, dft_stats.ssd_reuse_token_cnt);
        if (dft_reuse_token_cnt > 0) {
            printf("MTP total reuse token cnt: %d\n", dft_reuse_token_cnt);
        } else {
            printf("MTP no reuse token found\n");
        }
        lookup_tokens.resize(std::min<size_t>(lookup_tokens.size(), dft_reuse_token_cnt));
    }

    auto restore_stats = adptv.restore_kv_cache(lookup_tokens, {}, lora_adapters, 0, 0);
    printf("\nDram reuse token : %d \nCache reuse token : %d\n", restore_stats.dram_reuse_token_cnt, restore_stats.ssd_reuse_token_cnt);
    uint32_t total_reuse_token_cnt = restore_stats.dram_reuse_token_cnt + restore_stats.ssd_reuse_token_cnt;
    if (total_reuse_token_cnt > 0) {
        printf("Total reuse token cnt: %d\n", total_reuse_token_cnt);
    } else {
        printf("No reuse token found\n");
    }


    // prepare a batch for the prompt
    std::vector<llama_token> tokens_to_save;
    tokens_to_save.insert(tokens_to_save.end(), prompt_tokens.begin(), prompt_tokens.end());

    uint32_t skip = total_reuse_token_cnt;

    // MTP needs fresh nextn embd from ctx_tgt for pending_h; that only comes from a real
    // decode. Partial reuse still runs the prefill below and populates pending_h correctly.
    // Only full reuse (skip covers prompt[0..n_prompt-2]) skips prefill entirely, so roll
    // back one token to force a minimal prefill. Requires n_rs_seq >= 1 on hybrid archs.
    if (enable_mtp && skip > 0 && (int) skip == (int) n_prompt - 1) {
        if (llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (skip - 1), -1)) {
            skip -= 1;
        } else {
            fprintf(stderr, "%s: warning: seq_rm failed for boundary rollback; disabling KV reuse\n", __func__);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            skip = 0;
        }
    }

    // Keep the draft KV exactly `skip` long: the prefill mirrors tokens
    // [skip, n_prompt-1) into the draft context, and cells left behind at those
    // positions would be attended twice.
    if (adptv_dft && dft_reuse_token_cnt > skip) {
        if (!llama_memory_seq_rm(llama_get_memory(mtp.ctx_dft), 0, (llama_pos) skip, -1)) {
            fprintf(stderr, "%s: warning: seq_rm(draft) failed at p0=%u; disabling KV reuse\n", __func__, skip);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            llama_memory_seq_rm(llama_get_memory(mtp.ctx_dft), 0, -1, -1);
            skip = 0;
        }
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int64_t t_first_token_us = 0;
    int n_decode = 0;
    bool finished_by_eog = false;
    int n_drafted = 0;
    int n_accept  = 0;

    printf("Start decode: \n");

    if (!enable_mtp) {
        llama_batch batch = llama_batch_get_one(
            prompt_tokens.data() + skip,
            (int32_t) prompt_tokens.size() - (int32_t) skip);

        if (llama_model_has_encoder(model)) {
            if (llama_encode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }

            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }

            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }

        llama_token new_token_id;
        // KV cache already holds `skip` tokens, so n_pos starts there.
        for (int n_pos = (int) skip; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_vocab_is_eog(vocab, new_token_id)) {
                    finished_by_eog = true;
                    break;
                }

                if (t_first_token_us == 0) {
                    t_first_token_us = ggml_time_us();
                }
                print_token(vocab, new_token_id);

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);
                tokens_to_save.push_back(new_token_id);

                n_decode += 1;
            }
        }
    } else {
        // Prefill prompt except last token (id_last), then draft-mtp loop.
        // Ref: speculative-simple + common_speculative_impl_draft_mtp
        const int n_prefill_end = n_prompt - 1; // exclusive; keeps id_last out of KV
        llama_batch batch_tgt = llama_batch_init(n_batch_need, 0, 1);

        if ((int) skip < n_prefill_end) {
            batch_clear(batch_tgt);
            for (int i = (int) skip; i < n_prefill_end; ++i) {
                batch_add(batch_tgt, prompt_tokens[i], (llama_pos) i, 0, true);
            }
            if (llama_decode(ctx, batch_tgt) != 0) {
                fprintf(stderr, "%s: prompt prefill failed\n", __func__);
                mtp_free(mtp);
                return 1;
            }
            if (!mtp_process(mtp, batch_tgt)) {
                mtp_free(mtp);
                return 1;
            }
        }

        llama_token id_last = prompt_tokens.back();
        llama_pos   n_past  = (llama_pos) n_prefill_end;

        while (n_decode < n_predict) {
            // ids can be draft.size()+1 (all match + bonus). Cap draft so emitted
            // tokens never exceed remaining n_predict (e.g. -n 1 -> empty draft).
            const int remain       = n_predict - n_decode;
            const int draft_budget = std::max(0, remain - 1);
            const int draft_max_saved = mtp.n_draft_max;
            mtp.n_draft_max = std::min(draft_max_saved, draft_budget);

            std::vector<llama_token> draft;
            if (mtp.n_draft_max > 0) {
                draft = mtp_draft(mtp, id_last, n_past);
            }
            mtp.n_draft_max = draft_max_saved;
            n_drafted += (int) draft.size();

            llama_context * const ctx_draft_side = mtp.is_mem_shared ? ctx : mtp.ctx_dft;
            if (!llama_memory_seq_rm(llama_get_memory(ctx_draft_side), 0, n_past, -1)) {
                fprintf(stderr, "%s: seq_rm(draft side) failed at p0=%d - check n_rs_seq\n",
                        __func__, (int) n_past);
                break;
            }

            batch_clear(batch_tgt);
            batch_add(batch_tgt, id_last, n_past, 0, true);
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_add(batch_tgt, draft[i], n_past + 1 + (llama_pos) i, 0, true);
            }

            if (llama_decode(ctx, batch_tgt) != 0) {
                fprintf(stderr, "%s: target verify decode failed\n", __func__);
                break;
            }

            const std::vector<llama_token> ids = sample_and_accept_n(smpl, ctx, draft);
            const uint16_t n_accepted_draft = (uint16_t) (ids.size() - 1);
            n_accept += (int) n_accepted_draft;

            // Trim rejected drafts from target KV. Attention is causal, so verify's KV at
            // accepted positions equals what a fresh decode of just the accepted subset
            // would produce - no commit re-decode needed.
            if ((size_t) n_accepted_draft < draft.size()) {
                const llama_pos p0 = n_past + 1 + (llama_pos) n_accepted_draft;
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1)) {
                    fprintf(stderr, "%s: seq_rm(target) failed at p0=%d - check n_rs_seq\n",
                            __func__, (int) p0);
                    break;
                }
            }

            // Hand the accepted-subset batch to mtp_process:
            //   non-shared: it decodes into ctx_dft to catch draft KV up to accepted tokens
            //   all cases:  reads nextn embds from the verify pass (seq_rm above does not
            //               invalidate the ctx output buffer) into verify_h / pending_h
            batch_clear(batch_tgt);
            batch_add(batch_tgt, id_last, n_past, 0, true);
            for (uint16_t i = 0; i < n_accepted_draft; ++i) {
                batch_add(batch_tgt, ids[i], n_past + 1 + (llama_pos) i, 0, true);
            }
            if (!mtp_process(mtp, batch_tgt)) {
                break;
            }

            // id_last + accepted drafts are now in KV; ids.back() is the next input.
            n_past += (llama_pos) (1 + n_accepted_draft);

            bool   hit_eog = false;
            size_t i_eog   = ids.size();
            for (size_t i = 0; i < ids.size() && n_decode < n_predict; ++i) {
                if (llama_vocab_is_eog(vocab, ids[i])) {
                    finished_by_eog = true;
                    hit_eog = true;
                    i_eog = i;
                    break;
                }
                if (t_first_token_us == 0) {
                    t_first_token_us = ggml_time_us();
                }
                print_token(vocab, ids[i]);
                tokens_to_save.push_back(ids[i]);
                n_decode += 1;
            }

            id_last = ids.back();

            // EOG mid-batch: verify + seq_rm kept id_last and all accepted drafts in KV;
            // trim the tail past the EOG position so KV stays aligned with tokens_to_save.
            if (hit_eog && i_eog < n_accepted_draft) {
                const llama_pos p0 = n_past - (llama_pos) (n_accepted_draft - i_eog);
                if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1)) {
                    fprintf(stderr, "%s: warning: seq_rm(target) failed at p0=%d\n", __func__, (int) p0);
                }
                if (!mtp.is_mem_shared) {
                    if (!llama_memory_seq_rm(llama_get_memory(mtp.ctx_dft), 0, p0, -1)) {
                        fprintf(stderr, "%s: warning: seq_rm(draft) failed at p0=%d\n", __func__, (int) p0);
                    }
                }
                n_past = p0;
            }

            if (hit_eog) {
                break;
            }
        }

        llama_batch_free(batch_tgt);

        fprintf(stderr, "\nMTP: drafted %d, accepted %d, accept=%.1f%%\n",
                n_drafted, n_accept,
                n_drafted > 0 ? (100.0f * n_accept / n_drafted) : 0.0f);
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    // Server-style timings: TTFT = prompt processing (cache lookup + prefill + first sample);
    // TG throughput = tokens generated after first token / wall-clock generation window.
    const int pp_tokens = (int) n_prompt - (int) skip;
    const int cache_tokens = (int) skip;
    const double t_total_ms = (t_main_end - t_main_start) / 1e3;
    const double t_ttft_ms  = t_first_token_us > 0 ? (t_first_token_us - t_main_start) / 1e3 : 0.0;
    const double t_gen_ms   = (t_first_token_us > 0 && n_decode > 1) ? (t_main_end - t_first_token_us) / 1e3 : 0.0;
    const double pp_tps     = t_ttft_ms > 0 ? pp_tokens * 1e3 / t_ttft_ms : 0.0;
    const double tg_tps     = t_gen_ms  > 0 ? (n_decode - 1) * 1e3 / t_gen_ms : 0.0;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt      = %d tokens (%d cached, %d processed)\n", __func__, (int) n_prompt, cache_tokens, pp_tokens);
    fprintf(stderr, "%s: TTFT        = %.2f ms  (%.2f t/s over %d prompt tokens)\n", __func__, t_ttft_ms, pp_tps, pp_tokens);
    fprintf(stderr, "%s: TP          = %.2f t/s (%d generated tokens in %.2f ms after first)\n", __func__, tg_tps, std::max(0, n_decode - 1), t_gen_ms);
    fprintf(stderr, "%s: total       = %.2f ms, %d decoded tokens, %.2f t/s end-to-end\n",
            __func__, t_total_ms, n_decode,
            t_total_ms > 0 ? n_decode * 1e3 / t_total_ms : 0.0);

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    // build the kv-cache index for (prompt + generated) so the next run with
    // the same prefix can be served from the cache.
    //
    // On the n_predict-limit exit path the most recently sampled token was
    // prepared as the next decode input but never actually decoded into the KV
    // cache, so we trim one to keep tokens_to_save aligned with the KV cache.
    // On EOG exit `tokens_to_save` is already aligned (the EOG token was
    // never pushed), so no trim is needed.
    if (!tokens_to_save.empty()) {
        if (!finished_by_eog) {
            tokens_to_save.pop_back();
        }
        auto start = std::chrono::high_resolution_clock::now();
        if (!tokens_to_save.empty()) {
            adptv.save_kv_cache(
                tokens_to_save,
                /*mtmd_info     */ {},
                /*lora_adapters */ lora_adapters);
            // The draft context mirrors the target positions, so the same token
            // sequence keys both caches.
            if (adptv_dft) {
                adptv_dft->save_kv_cache(tokens_to_save, {}, {});
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Save kv cache time: " << duration.count() << " ms" << std::endl;
    }

    // flush any pending kv cache pages to disk before tearing down the model.
    adptv.flush_kv_cache();
    if (adptv_dft) {
        adptv_dft->flush_kv_cache();
        adptv_dft.reset(); // release the draft context before mtp_free() frees it
    }

    llama_sampler_free(smpl);
    mtp_free(mtp);
    llama_free(ctx);
    llama_model_free(model);

    printf("Clean temp caches\n");
    adptv.remove_owned_temp_caches();

    return 0;
}
