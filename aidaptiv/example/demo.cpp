//////////////////////////////////////////////////////
//
//  This example is modified from llama-simple
//
//////////////////////////////////////////////////////

#include "llama.h"
#include "aidaptiv.hpp"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -o offload_path [-n n_predict]\n"
           "        [-lora adapter.gguf] [-ls scale]\n"
           "        [-vc vram_experts_cached_gb] [-dc dram_experts_cached_gb]\n"
           "        [-sk ssd_kv_offload_gb] [-dk dram_kv_offload_gb]\n"
           "        [-kr kv_cache_resume_policy] [-fa]\n"
           "        [-f prompt_file] [-save-align N] [-no-print-prompt]\n"
           "        [prompt]\n", argv[0]);
    printf("\n");
}

static double us_to_ms(int64_t us) {
    return us / 1000.0;
}

static void print_timing(const char * name, int64_t us) {
    fprintf(stderr, "[perf] %-32s %.3f ms\n", name, us_to_ms(us));
}

static bool read_text_file(const std::string & path, std::string & text) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "error: unable to open prompt file '%s'\n", path.c_str());
        return false;
    }
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    ggml_time_init();

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of tokens to predict
    int n_predict = 32;
    std::string offload_path = "";
    int vram_experts_cached_gb = 0;
    int dram_experts_cached_gb = 0;

    // KV cache reuse parameters
    int  ssd_kv_offload_gb      = 0;      // -sk : SSD budget for kv cache (GB), -1 = auto
    int  dram_kv_offload_gb     = 0;      // -dk : DRAM budget for kv cache (GB), -1 = auto
    int  kv_cache_resume_policy = 1;      // -kr : 0 = no resume, 1 = try resume on startup
    bool flash_attn             = true;  // -fa : enable flash attention
    std::string lora_path;
    float       lora_scale      = 1.0f;
    std::string prompt_file;
    bool        print_prompt    = true;
    int         save_align      = 0;
    // parse command line arguments

    const auto t_total_start = ggml_time_us();

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
                    try {
                        offload_path = std::string(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
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
            } else if (strcmp(argv[i], "-f") == 0) {
                if (i + 1 < argc) {
                    prompt_file = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-save-align") == 0) {
                if (i + 1 < argc) {
                    try {
                        save_align = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-no-print-prompt") == 0) {
                print_prompt = false;
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
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
        if (!prompt_file.empty() && !read_text_file(prompt_file, prompt)) {
            return 1;
        }
    }

    // load dynamic backends

    const auto t_backend_start = ggml_time_us();
    ggml_backend_load_all();
    print_timing("backend_load_all", ggml_time_us() - t_backend_start);

    // init_log_and_param runs in the constructor (before model load).
    aidaptiv::setup_params sp;
    sp.ssd_kv_offload_gb      = ssd_kv_offload_gb;
    sp.dram_kv_offload_gb     = dram_kv_offload_gb;
    sp.kv_cache_resume_policy = static_cast<uint32_t>(kv_cache_resume_policy);
    sp.flash_attn             = flash_attn;
    sp.model_path             = model_path;

    fprintf(stderr, "[perf] config model='%s'\n", model_path.c_str());
    fprintf(stderr, "[perf] config prompt_file='%s' prompt_bytes=%zu n_predict=%d\n", prompt_file.c_str(), prompt.size(), n_predict);
    fprintf(stderr, "[perf] config offload='%s' vc=%d dc=%d sk=%d dk=%d kr=%d flash_attn=%d save_align=%d\n",
            offload_path.c_str(), vram_experts_cached_gb, dram_experts_cached_gb,
            ssd_kv_offload_gb, dram_kv_offload_gb, kv_cache_resume_policy, flash_attn ? 1 : 0, save_align);

    const auto t_aidaptiv_ctor_start = ggml_time_us();
    aidaptiv::Aidaptiv adptv(offload_path, offload_path, sp);
    print_timing("aidaptiv_ctor", ggml_time_us() - t_aidaptiv_ctor_start);
    printf("Aidaptiv Version: %s\n", adptv.version().c_str());

    // make sure to get right offload path from aidaptiv in hide ssd case.
    const std::string & resolved_offload = adptv.offload_path();
    fprintf(stderr, "[perf] resolved_offload='%s'\n", resolved_offload.c_str());
    
    // initialize the model

    // temp_uuid for MoE expert temp cache (model_params only; ctx does not need it
    // unless n_extend_ctx + SSD extend are enabled).
    const auto t_uuid_start = ggml_time_us();
    const uint64_t temp_uuid = adptv.generate_uuid();
    print_timing("aidaptiv_generate_uuid", ggml_time_us() - t_uuid_start);
    fprintf(stderr, "[perf] temp_uuid=%llu\n", (unsigned long long) temp_uuid);

    // Moe offload setting
    llama_model_params model_params = llama_model_default_params();
    model_params.temp_uuid              = temp_uuid;
    model_params.offload_folder         = resolved_offload.c_str();
    model_params.vram_experts_cached_gb = vram_experts_cached_gb;
    model_params.dram_experts_cached_gb = dram_experts_cached_gb;

    const auto t_model_load_start = ggml_time_us();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    print_timing("model_load", ggml_time_us() - t_model_load_start);

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
    const auto t_tokenize_count_start = ggml_time_us();
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    print_timing("tokenize_count", ggml_time_us() - t_tokenize_count_start);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    const auto t_tokenize_start = ggml_time_us();
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }
    print_timing("tokenize_prompt", ggml_time_us() - t_tokenize_start);
    fprintf(stderr, "[perf] prompt_tokens=%d ctx_tokens=%d batch_tokens=%d\n", n_prompt, n_prompt + n_predict - 1, n_prompt);

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    const auto t_context_init_start = ggml_time_us();
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    print_timing("context_init", ggml_time_us() - t_context_init_start);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const auto t_lora_apply_start = ggml_time_us();
    if (!apply_lora(ctx, lora_init, lora_adapters)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    print_timing("lora_apply", ggml_time_us() - t_lora_apply_start);

    const auto t_aidaptiv_init_start = ggml_time_us();
    adptv.init(ctx, model, lora_init, lora_adapters);
    print_timing("aidaptiv_init", ggml_time_us() - t_aidaptiv_init_start);

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    const auto t_print_prompt_start = ggml_time_us();
    if (print_prompt) {
        for (auto id : prompt_tokens) {
            char buf[128];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
        }
    }
    print_timing("print_prompt", ggml_time_us() - t_print_prompt_start);

    // always decode the last token to get fresh logits for sampling.
    std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
    const auto t_restore_start = ggml_time_us();
    auto restore_stats = adptv.restore_kv_cache(lookup_tokens, {}, lora_adapters, 0, 0);
    print_timing("restore_kv_cache", ggml_time_us() - t_restore_start);
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

    const auto t_batch_start = ggml_time_us();
    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data() + skip,
        (int32_t) prompt_tokens.size() - (int32_t) skip);
    print_timing("initial_batch_create", ggml_time_us() - t_batch_start);

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

    // main loop

    const auto t_main_start = ggml_time_us();
    int64_t t_prefill_us = 0;
    int64_t t_decode_us  = 0;
    int64_t t_sample_us  = 0;
    int64_t t_piece_us   = 0;
    int     n_prefill_tokens = 0;
    int n_decode = 0;
    llama_token new_token_id;
    bool finished_by_eog = false;
    bool first_decode = true;

    printf("Start decode: \n");
    // KV cache already holds `skip` tokens, so n_pos starts there.
    for (int n_pos = (int) skip; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        const int batch_tokens = batch.n_tokens;
        const auto t_eval_start = ggml_time_us();
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        const int64_t eval_us = ggml_time_us() - t_eval_start;
        if (first_decode) {
            t_prefill_us += eval_us;
            n_prefill_tokens += batch_tokens;
            print_timing("prefill_decode_call", eval_us);
            fprintf(stderr, "[perf] prefill_tokens=%d reused_tokens=%u remaining_prompt_tokens=%d\n",
                    batch_tokens, total_reuse_token_cnt, (int) prompt_tokens.size() - (int) skip);
            first_decode = false;
        } else {
            t_decode_us += eval_us;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            const auto t_sample_start = ggml_time_us();
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            t_sample_us += ggml_time_us() - t_sample_start;

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                finished_by_eog = true;
                break;
            }

            char buf[128];
            const auto t_piece_start = ggml_time_us();
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            t_piece_us += ggml_time_us() - t_piece_start;
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
            tokens_to_save.push_back(new_token_id);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    print_timing("prefill_total", t_prefill_us);
    fprintf(stderr, "[perf] prefill_tokens_total=%d prefill_ms_per_token=%.3f\n",
            n_prefill_tokens, n_prefill_tokens > 0 ? us_to_ms(t_prefill_us) / n_prefill_tokens : 0.0);
    print_timing("generation_decode_total", t_decode_us);
    print_timing("sampling_total", t_sample_us);
    print_timing("token_to_piece_total", t_piece_us);
    print_timing("main_loop_total", t_main_end - t_main_start);

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "[perf] moe_hit single_vram=%.6f single_dram=%.6f multi_vram=%.6f multi_dram=%.6f total_vram=%.6f total_dram=%.6f\n",
            llama_moe_get_single_token_vram_hit_rate(ctx),
            llama_moe_get_single_token_dram_hit_rate(ctx),
            llama_moe_get_multi_token_vram_hit_rate(ctx),
            llama_moe_get_multi_token_dram_hit_rate(ctx),
            llama_moe_get_vram_hit_rate(ctx),
            llama_moe_get_dram_hit_rate(ctx));

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
        if (save_align > 0) {
            const size_t before = tokens_to_save.size();
            tokens_to_save.resize((before / (size_t) save_align) * (size_t) save_align);
            fprintf(stderr, "[perf] save_align=%d save_tokens_before_align=%zu save_tokens_after_align=%zu\n",
                    save_align, before, tokens_to_save.size());
        }
        if (!tokens_to_save.empty()) {
            const auto t_save_start = ggml_time_us();
            adptv.save_kv_cache(
                tokens_to_save,
                /*mtmd_info     */ {},
                /*lora_adapters */ lora_adapters);
            print_timing("save_kv_cache", ggml_time_us() - t_save_start);
            fprintf(stderr, "[perf] save_tokens=%zu finished_by_eog=%d\n", tokens_to_save.size(), finished_by_eog ? 1 : 0);
        }
    }

    // flush any pending kv cache pages to disk before tearing down the model.
    const auto t_flush_start = ggml_time_us();
    adptv.flush_kv_cache();
    print_timing("flush_kv_cache", ggml_time_us() - t_flush_start);

    const auto t_cleanup_start = ggml_time_us();
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    print_timing("llama_cleanup", ggml_time_us() - t_cleanup_start);

    printf("Clean temp caches\n");
    const auto t_remove_temp_start = ggml_time_us();
    adptv.remove_temp_caches();
    print_timing("remove_temp_caches", ggml_time_us() - t_remove_temp_start);
    print_timing("total_process", ggml_time_us() - t_total_start);

    return 0;
}
