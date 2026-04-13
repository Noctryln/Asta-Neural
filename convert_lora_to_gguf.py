import argparse
import os
import json
import torch
from safetensors.torch import load_file
import gguf
import numpy as np
import re

def get_gguf_name(name):
    mapping = {
        "q_proj":    "attn_q",
        "k_proj":    "attn_k",
        "v_proj":    "attn_v",
        "o_proj":    "attn_output",
        "gate_proj": "ffn_gate",
        "up_proj":   "ffn_up",
        "down_proj": "ffn_down",
    }

    # Normalisasi prefix PEFT — buang apapun sebelum "layers."
    # Handles: base_model.model.model.layers.X, model.layers.X, dll
    match = re.search(r"layers\.(\d+)\.([\w.]+)\.(lora_[AB])\.weight$", name)
    if not match:
        return None

    layer_idx = match.group(1)
    module_path = match.group(2) 
    lora_type = "lora_a" if match.group(3) == "lora_A" else "lora_b"

    # Ambil bagian terakhir dari module_path sebagai proj name
    proj_name = module_path.split(".")[-1]

    target = mapping.get(proj_name)
    if not target:
        return None

    return f"blk.{layer_idx}.{target}.weight.{lora_type}"

def convert_lora(input_path, output_path, alpha=None):
    print(f"[*] Loading adapter from {input_path}")
    tensors = load_file(input_path)
    
    if alpha is None:
        config_path = os.path.join(os.path.dirname(input_path), "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                alpha = cfg.get("lora_alpha", 32)
                print(f"[*] Found alpha in config: {alpha}")
        else:
            alpha = 32
            print(f"[!] No adapter_config.json found, using default alpha: {alpha}")

    writer = gguf.GGUFWriter(output_path, "qwen3")
    
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_string("general.architecture", "qwen3")
    writer.add_string("general.name", "Asta Qwen3 LoRA")
    
    writer.add_float32("adapter.lora.alpha", float(alpha))
    
    count = 0
    skipped = []
    for name, tensor in tensors.items():
        new_name = get_gguf_name(name)
        if new_name:
            data = tensor.to(torch.float32).numpy()
            writer.add_tensor(new_name, data)
            count += 1
        else:
            skipped.append(name)
    
    if skipped:
        print(f"[!] {len(skipped)} tensor diskip (tidak ada mapping):")
        for s in skipped:
            print(f"    - {s}")

    print(f"[*] Writing GGUF file to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Converted {count} tensors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to adapter_model.safetensors")
    parser.add_argument("output", help="Path to output .gguf")
    parser.add_argument("--alpha", type=float, help="LoRA alpha")
    
    args = parser.parse_args()
    convert_lora(args.input, args.output, args.alpha)
