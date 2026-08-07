// Copyright (c) 2026 SteerLab prototype.
// SPDX-License-Identifier: BSD-3-Clause
//
// Thin JNI shim over the FORKED GenieX public C ABI (geniex.h) — the exact
// boundary an ordinary application would use. Nothing here talks to
// geniex-qairt internals; runtime steering flows through
// geniex_LlmGenerateInput.aux_inputs and diagnostics through
// geniex_llm_get_aux_stats.

#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "geniex.h"

namespace {

geniex_LLM* g_llm = nullptr;
std::mutex  g_mutex;  // GenieX handles are not thread-safe; serialize requests
int64_t     g_context_loads = 0;

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    env->ReleaseStringUTFChars(s, c);
    return out;
}

std::string jsonEscape(const std::string& in) {
    std::string out;
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) break;
                out += c;
        }
    }
    return out;
}

std::string statsJson() {
    geniex_LlmAuxStats s{};
    if (g_llm && geniex_llm_get_aux_stats(g_llm, &s) == GENIEX_SUCCESS) {
        std::ostringstream os;
        os << "{\"prefill_writes\":" << s.prefill_writes << ",\"decode_writes\":" << s.decode_writes
           << ",\"aux_requests\":" << s.aux_requests << ",\"total_requests\":" << s.total_requests
           << ",\"context_loads\":" << g_context_loads << "}";
        return os.str();
    }
    return "{}";
}

}  // namespace

extern "C" {

// init(nativeLibDir, bundleDir) -> JSON {ok, load_ms, error?, stats}
JNIEXPORT jstring JNICALL Java_com_dragonnest_geniexsteeringlab_GenieXBridge_init(
    JNIEnv* env, jclass, jstring native_lib_dir, jstring bundle_dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string lib_dir = jstr(env, native_lib_dir);
    const std::string bundle  = jstr(env, bundle_dir);

    // Plugin discovery + QNN backend paths both key off GENIEX_PLUGIN_PATH on
    // Android (flat APK lib dir); the DSP skel is found via ADSP_LIBRARY_PATH.
    setenv("GENIEX_PLUGIN_PATH", lib_dir.c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", lib_dir.c_str(), 1);

    std::ostringstream os;
    const auto t0 = std::chrono::steady_clock::now();

    if (!g_llm) {
        static bool inited = false;
        if (!inited) {
            // Surface every runtime log line in logcat under "SteerLabGenieX".
            geniex_set_log([](geniex_LogLevel level, const char* msg) {
                __android_log_print(
                    level >= GENIEX_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "SteerLabGenieX", "%s",
                    msg ? msg : "");
            });
            const int32_t rc = geniex_init();
            if (rc != GENIEX_SUCCESS) {
                os << "{\"ok\":false,\"error\":\"geniex_init failed: " << rc << "\"}";
                return env->NewStringUTF(os.str().c_str());
            }
            // APK jniLibs are flat, but Registry::scan_plugins expects
            // <plugin_path>/<name>/libgeniex_plugin.so subdirectories, so
            // register the qairt plugin manually through the public ABI.
            const std::string plugin_so = lib_dir + "/libgeniex_plugin.so";
            void*             handle    = dlopen(plugin_so.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) {
                os << "{\"ok\":false,\"error\":\"dlopen plugin: " << jsonEscape(dlerror() ? dlerror() : "?") << "\"}";
                return env->NewStringUTF(os.str().c_str());
            }
            auto id_fn     = reinterpret_cast<geniex_plugin_id_func>(dlsym(handle, "plugin_id"));
            auto create_fn = reinterpret_cast<geniex_create_plugin_func>(dlsym(handle, "create_plugin"));
            if (!id_fn || !create_fn || geniex_register_plugin(id_fn, create_fn) != GENIEX_SUCCESS) {
                os << "{\"ok\":false,\"error\":\"plugin symbol/registration failure\"}";
                return env->NewStringUTF(os.str().c_str());
            }
            inited = true;
        }

        geniex_LlmCreateInput in{};
        const std::string model_path = bundle + "/genie_config.json";
        in.model_name  = "steeringlab-qwen";
        in.model_path  = model_path.c_str();
        in.plugin_id   = "qairt";
        in.device_id   = nullptr;
        const int32_t rc = geniex_llm_create(&in, &g_llm);
        if (rc != GENIEX_SUCCESS || !g_llm) {
            os << "{\"ok\":false,\"error\":\"geniex_llm_create failed: " << rc << " ("
               << jsonEscape(geniex_get_error_message(static_cast<geniex_ErrorCode>(rc))) << ")\"}";
            g_llm = nullptr;
            return env->NewStringUTF(os.str().c_str());
        }
        ++g_context_loads;
    }

    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    os << "{\"ok\":true,\"load_ms\":" << ms << ",\"stats\":" << statsJson() << "}";
    return env->NewStringUTF(os.str().c_str());
}

// generate(prompt, alpha, vector[], useAux, maxTokens) -> JSON
JNIEXPORT jstring JNICALL Java_com_dragonnest_geniexsteeringlab_GenieXBridge_generate(
    JNIEnv* env, jclass, jstring prompt, jfloat alpha, jfloatArray vec, jboolean use_aux, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ostringstream os;
    if (!g_llm) {
        return env->NewStringUTF("{\"ok\":false,\"error\":\"model not loaded\"}");
    }

    // Chat formatting through the stock ABI.
    const std::string user_text = jstr(env, prompt);
    geniex_LlmChatMessage msg{"user", user_text.c_str()};
    geniex_LlmApplyChatTemplateInput tin{};
    tin.messages              = &msg;
    tin.message_count         = 1;
    tin.add_generation_prompt = true;
    geniex_LlmApplyChatTemplateOutput tout{};
    std::string formatted = user_text;
    if (geniex_llm_apply_chat_template(g_llm, &tin, &tout) == GENIEX_SUCCESS && tout.formatted_text) {
        formatted = tout.formatted_text;
        geniex_free(tout.formatted_text);
    }

    geniex_SamplerConfig sampler{};
    sampler.temperature = 0.0f;  // deterministic comparisons across alphas
    sampler.top_p       = 1.0f;
    sampler.seed        = 42;
    geniex_GenerationConfig cfg{};
    cfg.max_tokens     = max_tokens > 0 ? max_tokens : 96;
    cfg.sampler_config = &sampler;

    geniex_LlmGenerateInput in{};
    in.prompt_utf8 = formatted.c_str();
    in.config      = &cfg;

    // Runtime steering: alpha [1] + steering_vector [1,1,H] as aux inputs.
    std::vector<float>              vec_data;
    std::vector<geniex_NamedTensor> aux;
    const int64_t alpha_dims[1] = {1};
    int64_t       vec_dims[3]   = {1, 1, 0};
    float         alpha_val     = alpha;
    if (use_aux) {
        aux.push_back({"alpha", GENIEX_TENSOR_DTYPE_FLOAT32, alpha_dims, 1, &alpha_val, sizeof(float)});
        if (vec) {
            const jsize n = env->GetArrayLength(vec);
            vec_data.resize(static_cast<size_t>(n));
            env->GetFloatArrayRegion(vec, 0, n, vec_data.data());
            vec_dims[2] = n;
            aux.push_back({"steering_vector", GENIEX_TENSOR_DTYPE_FLOAT32, vec_dims, 3, vec_data.data(),
                vec_data.size() * sizeof(float)});
        }
        in.aux_inputs      = aux.data();
        in.aux_input_count = static_cast<int32_t>(aux.size());
    }

    geniex_LlmAuxStats before{};
    geniex_llm_get_aux_stats(g_llm, &before);

    geniex_LlmGenerateOutput out{};
    const auto    t0 = std::chrono::steady_clock::now();
    const int32_t rc = geniex_llm_generate(g_llm, &in, &out);
    const auto    ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    geniex_LlmAuxStats after{};
    geniex_llm_get_aux_stats(g_llm, &after);

    // Multi-turn KV reuse is not what we want for A/B alpha comparisons —
    // reset so every request scores the same prompt from a clean cache.
    geniex_llm_reset(g_llm);

    if (rc != GENIEX_SUCCESS) {
        os << "{\"ok\":false,\"error\":\"generate failed: " << rc << " ("
           << jsonEscape(geniex_get_error_message(static_cast<geniex_ErrorCode>(rc))) << ")\",\"stats\":"
           << statsJson() << "}";
        return env->NewStringUTF(os.str().c_str());
    }

    os << "{\"ok\":true,\"text\":\"" << jsonEscape(out.full_text ? out.full_text : "") << "\"" << ",\"ms\":" << ms
       << ",\"prompt_tokens\":" << out.profile_data.prompt_tokens
       << ",\"generated_tokens\":" << out.profile_data.generated_tokens
       << ",\"tps\":" << out.profile_data.decoding_speed
       << ",\"delta_prefill_writes\":" << (after.prefill_writes - before.prefill_writes)
       << ",\"delta_decode_writes\":" << (after.decode_writes - before.decode_writes)
       << ",\"stats\":" << statsJson() << "}";
    if (out.full_text) geniex_free(out.full_text);
    return env->NewStringUTF(os.str().c_str());
}

}  // extern "C"
