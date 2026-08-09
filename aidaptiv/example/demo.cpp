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
           "        [prompt]\n", argv[0]);
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

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
    // parse command line arguments

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
        if (model_path.empty() || offload_path.empty()) {
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
    }

    // load dynamic backends

    ggml_backend_load_all();

    // init_log_and_param runs in the constructor (before model load).
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

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
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

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

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

    // always decode the last token to get fresh logits for sampling.
    std::vector<llama_token> lookup_tokens(prompt_tokens.begin(), prompt_tokens.end() - 1);
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

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    bool finished_by_eog = false;

    printf("Start decode: \n");
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

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
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

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

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
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Save kv cache time: " << duration.count() << " ms" << std::endl;
    }

    // flush any pending kv cache pages to disk before tearing down the model.
    adptv.flush_kv_cache();

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("Clean temp caches\n");
    adptv.remove_owned_temp_caches();

    return 0;
}
