# Cek apakah LoRA benar-benar mengubah output vs base model
from llama_cpp import Llama

TEST_PROMPT = (
    "<|im_start|>system\n"
    "Namaku Asta.<|im_end|>\n"
    "<|im_start|>user\n"
    ">>> INPUT BARU <<<\n"
    "\"aku kangen kamu\"\n"
    "---\n"
    "ANALISIS:\n\n"
    "=== STEP 1: PERCEPTION ===\n"
    "TOPIC:\n"
    "SENTIMENT:\n"
    "URGENCY:\n"
    "HIDDEN_NEED:<|im_end|>\n"
    "<|im_start|>assistant\n"
    "=== STEP 1: PERCEPTION ==="
)

# Test 1: tanpa LoRA
llm_base = Llama(model_path="./model/Qwen3-4B-2507/Qwen3-4B-2507.gguf", n_gpu_layers=-1, verbose=False)
result_base = llm_base.create_completion(
    prompt=TEST_PROMPT, max_tokens=150,
    temperature=0.1, stop=["STOP", "<|im_end|>"], echo=False,
)
print("=== BASE (tanpa LoRA) ===")
print(result_base["choices"][0]["text"])

# Test 2: dengan LoRA
llm_lora = Llama(
    model_path="./model/Qwen3-4B-2507/Qwen3-4B-2507.gguf",
    lora_path="model/LoRA-all-adapter/thought_v4.gguf",
    lora_scale=1.0,
    n_gpu_layers=-1, verbose=False,
)
result_lora = llm_lora.create_completion(
    prompt=TEST_PROMPT, max_tokens=150,
    temperature=0.1, stop=["STOP", "<|im_end|>"], echo=False,
)
print("=== DENGAN LoRA ===")
print(result_lora["choices"][0]["text"])