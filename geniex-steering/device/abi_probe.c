// Minimal C-ABI reproduction of the SteerLab JNI flow, runnable via adb shell
// so stderr/logs are visible. Usage: abi_probe <plugin_dir> <bundle_dir> [alpha]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "geniex.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <plugin_dir> <bundle_dir> [alpha]\n", argv[0]);
        return 2;
    }
    setenv("GENIEX_PLUGIN_PATH", argv[1], 1);
    setenv("ADSP_LIBRARY_PATH", argv[1], 1);

    fprintf(stderr, "[abi_probe] geniex_init...\n");
    int32_t rc = geniex_init();
    fprintf(stderr, "[abi_probe] geniex_init rc=%d\n", rc);
    if (rc != 0) return 1;

    geniex_GetPluginListOutput plugins = {0};
    if (geniex_get_plugin_list(&plugins) == 0) {
        for (int i = 0; i < plugins.plugin_count; ++i)
            fprintf(stderr, "[abi_probe] plugin[%d]=%s\n", i, plugins.plugin_ids[i]);
    }

    char model_path[1024];
    snprintf(model_path, sizeof model_path, "%s/genie_config.json", argv[2]);
    geniex_LlmCreateInput in = {0};
    in.model_name = "abi-probe";
    in.model_path = model_path;
    in.plugin_id  = "qairt";

    geniex_LLM* llm = NULL;
    fprintf(stderr, "[abi_probe] geniex_llm_create...\n");
    rc = geniex_llm_create(&in, &llm);
    fprintf(stderr, "[abi_probe] geniex_llm_create rc=%d llm=%p\n", rc, (void*)llm);
    if (rc != 0 || !llm) return 1;

    float alpha = (argc > 3) ? (float)atof(argv[3]) : 0.0f;
    const int64_t adims[1] = {1};
    geniex_NamedTensor aux[1] = {{"alpha", GENIEX_TENSOR_DTYPE_FLOAT32, adims, 1, &alpha, sizeof alpha}};

    geniex_SamplerConfig sampler = {0};
    sampler.temperature = 0.0f;
    geniex_GenerationConfig cfg = {0};
    cfg.max_tokens     = 16;
    cfg.sampler_config = &sampler;

    geniex_LlmGenerateInput gin = {0};
    gin.prompt_utf8 = "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    gin.config      = &cfg;
    if (argc > 3) {
        gin.aux_inputs      = aux;
        gin.aux_input_count = 1;
    }

    geniex_LlmGenerateOutput out = {0};
    fprintf(stderr, "[abi_probe] geniex_llm_generate (alpha=%s)...\n", argc > 3 ? argv[3] : "off");
    rc = geniex_llm_generate(llm, &gin, &out);
    fprintf(stderr, "[abi_probe] generate rc=%d text=%s\n", rc, out.full_text ? out.full_text : "(null)");

    geniex_LlmAuxStats stats;
    if (geniex_llm_get_aux_stats(llm, &stats) == 0) {
        fprintf(stderr, "[abi_probe] aux stats: prefill=%lld decode=%lld reqs=%lld/%lld\n",
            (long long)stats.prefill_writes, (long long)stats.decode_writes, (long long)stats.aux_requests,
            (long long)stats.total_requests);
    }
    geniex_llm_destroy(llm);
    return rc == 0 ? 0 : 1;
}
