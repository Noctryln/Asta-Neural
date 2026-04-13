import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

DEFAULT_CONFIG = {
    "model_choice":             "2",
    "device":                   "cpu",
    "use_lora":                 False,
    "lora_n_gpu_layers":        0,
    "memory_mode":              "hybrid",
    "web_search_enabled":       True,
    "n_batch":                  1024,
    "tavily_api_key":           "",
    "serper_api_key":           "",
    "internal_thought_enabled": True,
    "long_thinking_enabled":    False,
    "long_thinking_max_tokens": 1536,
    "use_dynamic_prompt":       True,
    "thought_n_ctx":            3072,
    "thought_max_tokens":       1024,
    "thought_reset_every":      10,
    "use_model_thought_logic":  True,
    "separate_thought_model":   True,
    "token_budget": {
        "total_ctx":         8192,
        "response_reserved": 512,
        "system_identity":   350,
        "memory_budget":     600,
    },
}

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        if isinstance(data.get("token_budget"), dict):
            merged["token_budget"] = {**DEFAULT_CONFIG["token_budget"], **data["token_budget"]}
        return merged
    except (json.JSONDecodeError, ValueError):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    try:
        from engine.web_tools import invalidate_cfg_cache
        invalidate_cfg_cache()
    except ImportError:
        pass

def setup_wizard(cfg: dict) -> dict:
    print("\n" + "=" * 50)
    print("  ASTA — Setup Awal (hanya sekali)")
    print("=" * 50)

    print("\nPilih model response:")
    print("  1. Qwen3 4B (lebih ringan)")
    print("  2. Sailor2 8B (lebih pintar) [default]")
    choice = input("Pilihan (default = 2): ").strip() or "2"
    cfg["model_choice"] = choice if choice in ("1", "2") else "2"

    print("\nPilih device:")
    print("  1. CPU [default]")
    print("  2. GPU CUDA")
    dev = input("Pilihan (default = 1): ").strip()
    cfg["device"] = "gpu" if dev == "2" else "cpu"

    use_lora = input("\nGunakan LoRA adapter? (y/n, default = n): ").strip().lower()
    cfg["use_lora"] = use_lora == "y"
    if cfg["use_lora"]:
        try:
            lg = input("Layer LoRA di GPU? (default = 0): ").strip()
            cfg["lora_n_gpu_layers"] = int(lg) if lg else 0
        except ValueError:
            cfg["lora_n_gpu_layers"] = 0

    ws = input("\nAktifkan web search? (y/n, default = y): ").strip().lower()
    cfg["web_search_enabled"] = ws != "n"
    if cfg["web_search_enabled"]:
        tavily_key = input("  Tavily API key (kosong = skip): ").strip()
        cfg["tavily_api_key"] = tavily_key
        if not tavily_key:
            serper_key = input("  Serper API key (kosong = pakai DDG+Wikipedia): ").strip()
            cfg["serper_api_key"] = serper_key

    it = input("Aktifkan internal thought? (y/n, default = y): ").strip().lower()
    cfg["internal_thought_enabled"] = it != "n"

    if cfg["internal_thought_enabled"]:
        lt = input("Aktifkan long thinking? (y/n, default = n): ").strip().lower()
        cfg["long_thinking_enabled"] = lt == "y"

    if cfg["internal_thought_enabled"] and cfg["model_choice"] == "2":
        sep = input("\nGunakan model thought terpisah (3B)? (y/n, default = n): ").strip().lower()
        cfg["separate_thought_model"] = sep == "y"
        if cfg["separate_thought_model"]:
            try:
                nc = input("  thought_n_ctx (default = 3072): ").strip()
                cfg["thought_n_ctx"] = int(nc) if nc else 3072
            except ValueError:
                cfg["thought_n_ctx"] = 3072

    save_config(cfg)
    print("\nKonfigurasi disimpan ke config.json")
    print("Untuk reset: python core.py --setup\n")
    return cfg
