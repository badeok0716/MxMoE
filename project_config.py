import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = f"{CUR_DIR}/calib"

ID2NAME = {
    "ds2": "deepseek-ai/DeepSeek-V2-Lite",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2_moe": "Qwen/Qwen1.5-MoE-A2.7B",
    "qwen2_moe_57b": "Qwen/Qwen2-57B-A14B-Instruct",
}

EXPERT_QUANT_LOSS = {
    "rtn": {
        # "ds2": {
        #     "w8a8_g-1_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a4_g-1_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a4_g128_sym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
        #     "w4a16_g-1_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-model_out_norm.json",
        #     "w4a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        #     "w3a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        #     "w2a16_g128_asym": f"{CALIB_DIR}/ds2-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        # },
        **{
            k: {
                "w8a8_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W8A8_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g128_sym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A4_g128_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W8A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w8a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w3a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w2a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w1a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-rtn-W1A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
            }
            for k in ["qwen2_moe", "mixtral", "qwen2_moe_57b", "ds2"]
        },
    },
    "gptq": {
        "qwen2_moe": {
            "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W4A16_g-1_asym-wiki2-128-4096-model_out_norm.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W4A16_g128_asym-wiki2-128-4096-model_out_norm.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W3A16_g128_asym-wiki2-128-4096-model_out_norm.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe/qwen2_moe-MOE-gptq-W2A16_g128_asym-wiki2-128-4096-model_out_norm.json",
        },
    },
    "gptq-had": {
        **{
            k: {
                "w8a8_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g-1_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A4_g-1_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a4_g128_sym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w8a16_g-1_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W8A16_g-1_asym-wiki2-128-4096-layer_out_norm.json",
                "w4a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W4A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w3a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W3A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w2a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
                "w1a16_g128_asym": f"{CALIB_DIR}/{k}/{k}-MOE-gptq-had-W1A16_g128_asym-wiki2-128-4096-layer_out_norm.json",
            }
            for k in ["qwen2_moe", "mixtral", "qwen2_moe_57b", "ds2"]
        },
    },
    "rtn-fisher": {
        "qwen2_moe": {
            # "w8a8_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g128_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
            "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-fisher.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A16_g128_asym-wiki2-128-4096-fisher.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W3A16_g128_asym-wiki2-128-4096-fisher.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-fisher.json",
        },
    },
    "gptq-fisher": {
        "qwen2_moe": {
            # "w8a8_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W8A8_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g-1_sym": f"{CALIB_DIR}/qwen2_moe-MOE-rtn-W4A4_g-1_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a4_g128_sym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-had-W4A4_g128_sym-wiki2-128-4096-model_out_norm.json",
            # "w4a16_g-1_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W4A16_g-1_asym-wiki2-128-4096-fisher.json",
            "w4a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W4A16_g128_asym-wiki2-128-4096-fisher.json",
            "w3a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W3A16_g128_asym-wiki2-128-4096-fisher.json",
            "w2a16_g128_asym": f"{CALIB_DIR}/qwen2_moe-MOE-gptq-W2A16_g128_asym-wiki2-128-4096-fisher.json",
        },
    },
}
