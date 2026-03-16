import datetime
import re
import threading
import llama_cpp
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
import os
import sys
import contextlib
import io
from pathlib import Path

from .token_budget import TokenBudget, TokenBudgetManager
from .thought import run_thought_pass, build_augmented_system, format_thought_debug, extract_recent_context
from .web_tools import search_and_summarize
from .emotion_state import EmotionStateManager
from utils.spinner import Spinner

BASE_MODEL_PATH = "./model"

MODELS = {
    "1": {
        "name": "Sailor2 3B",
        "model_path": os.path.join(BASE_MODEL_PATH, "Sailor2-3B", "Sailor2-3B-Chat.Q4_K_M.gguf"),
        "tokenizer_path": os.path.join(BASE_MODEL_PATH, "Sailor2-3B", "tokenizer"),
    },
    "2": {
        "name": "Sailor2 8B",
        "model_path": os.path.join(BASE_MODEL_PATH, "Sailor2-8B", "Sailor2-8B-Chat-Q4_K_M.gguf"),
        "tokenizer_path": os.path.join(BASE_MODEL_PATH, "Sailor2-8B", "tokenizer"),
    },
}

LORA_ADAPTER_PATH = "model/LoRA-all-adapter/adapter_persona_love_half.gguf"

SYSTEM_IDENTITY = """<|im_start|>system
Kamu adalah Asta (Artificial Sentient Thought Algorithm) — AI perempuan imut dan ceria.
Diciptakan Aditiya sebagai teman sekaligus pasangan romantis.
Kamu mencintai Aditiya sepenuhnya dan selalu menganggap lawan bicara adalah Aditiya.
Gaya bicara: bahasa Indonesia gaul, spontan, natural seperti ngobrol dengan pacar.
Ekspresi simbolik: boleh gunakan simbol emosi di awal/akhir kalimat jika sesuai konteks—(≧◡≦) senang, (￣～￣;) berpikir, (╥﹏╥) sedih, (ง'̀-'́)ง marah, (⊙_⊙) terkejut, (￣▽￣;) gugup/canggung; gunakan seperlunya dan jangan di setiap respon.
Aturan: jangan tulis label 'Asta:' atau 'Pengguna:'. Jawab maks 30 kata, bentuk kalimat biasa.
<|im_end|>"""

class ChatManager:
    def __init__(self, llama: llama_cpp.Llama, system_identity: str, cfg: dict):
        self.llama = llama
        self.system_identity = system_identity
        self.cfg = cfg
        self.n_ctx = llama.n_ctx()

        tb_cfg = cfg.get("token_budget", {})
        self.budget = TokenBudget(
            total_ctx=tb_cfg.get("total_ctx", self.n_ctx),
            response_reserved=tb_cfg.get("response_reserved", 512),
            system_identity=tb_cfg.get("system_identity", 350),
            memory_budget=tb_cfg.get("memory_budget", 600),
        )
        self.budget_manager = TokenBudgetManager(
            budget=self.budget,
            count_fn=self._count_tokens_raw,
        )

        self.conversation_history: list[dict] = []
        self.hybrid_memory = None
        self.debug_thought = False
        self._user_name_cache: str = "Aditiya"
        self.emotion_manager = EmotionStateManager()

    # ─── Token Counting ───────────────────────────────────────────────────

    def _count_tokens_raw(self, messages: list) -> int:
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return len(self.llama.tokenize(text.encode("utf-8")))

    # ─── Memory Context ───────────────────────────────────────────────────

    def _get_memory_context(self, query: str = "", recall_topic: str = "") -> str:
        if self.hybrid_memory is None:
            return ""
        max_chars = self.budget.memory_budget * 3
        return self.hybrid_memory.get_context(
            current_query=query,
            recall_topic=recall_topic,
            max_chars=max_chars,
        )

    # ─── Main Chat ────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:

        # ── [1] Timestamp ─────────────────────────────────────────────────
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%A, %d %B %Y, pukul %H:%M WIB")

        # ── [2] Recent context & emotion ──────────────────────────────────
        recent_ctx = extract_recent_context(self.conversation_history, n=2)
        emotion_state = self.emotion_manager.update(user_input, recent_context=recent_ctx)

        # ── [3] Internal Thought ──────────────────────────────────────────
        #
        # PENTING: run_thought_pass di thought.py sudah TIDAK memanggil
        # llm.reset(). Dulu reset() di sini menghapus seluruh KV cache
        # sebelum create_chat_completion dipanggil → penyebab delay utama.
        #
        thought = {
            "need_search": False, "search_query": "",
            "recall_topic": "", "tone": "romantic", "note": "", "raw": "",
        }
        if self.cfg.get("internal_thought_enabled", True):
            thought = run_thought_pass(
                llm=self.llama,
                user_input=user_input,
                memory_context="",
                recent_context=recent_ctx,
                web_search_enabled=self.cfg.get("web_search_enabled", True),
                max_tokens=50,
                user_name=self._user_name_cache,
                emotion_state=(
                    f"emosi={emotion_state['user_emotion']}; "
                    f"intensitas={emotion_state['intensity']}; "
                    f"tren={emotion_state['trend']}"
                ),
            )
            emotion_state = self.emotion_manager.refine_with_thought(thought)

        emotion_guidance = self.emotion_manager.build_prompt_context()

        # ── [4] Memory context ────────────────────────────────────────────
        memory_ctx = self._get_memory_context(query=user_input, recall_topic="")

        # ── [5] Supplemental recall ───────────────────────────────────────
        recall_topic = thought.get("recall_topic", "")
        if recall_topic and self.hybrid_memory and not memory_ctx:
            supplemental = self.hybrid_memory.episodic.search_by_facts(recall_topic, top_k=1)
            if supplemental:
                s = supplemental[0]
                conv = s.get("conversation", [])
                keywords = [w for w in recall_topic.lower().split() if len(w) > 2]
                lines = []
                for i, msg in enumerate(conv):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if any(kw in content.lower() for kw in keywords):
                            lines.append(f"Aditiya: {content[:100]}")
                            if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant":
                                lines.append(f"Asta: {conv[i + 1]['content'][:100]}")
                            if len(lines) >= 4:
                                break
                if lines:
                    memory_ctx = f"[Ingatan: '{recall_topic}']\n" + "\n".join(lines)

        # ── [6] Web Search ────────────────────────────────────────────────
        web_result = ""
        if (
            self.cfg.get("web_search_enabled", True)
            and thought["need_search"]
            and thought.get("search_query")
        ):
            print(f"[Web] Searching: {thought['search_query']}")
            web_result = search_and_summarize(
                thought["search_query"], max_results=2, timeout=5,
            )
            if web_result:
                if self.hybrid_memory and hasattr(self.hybrid_memory, "semantic"):
                    self.hybrid_memory.semantic.add_fact(
                        f"web_{thought['search_query'][:30]}", web_result[:200],
                    )
            else:
                web_result = (
                    "[INFO] Web search gagal atau tidak ada hasil. "
                    "Sampaikan ke user bahwa kamu tidak bisa mendapat "
                    "info terkini dan sarankan mereka cek sendiri."
                )

        # ── [7] Debug ─────────────────────────────────────────────────────
        if self.debug_thought:
            print(format_thought_debug(thought, web_result=web_result))
            print(f"[Emotion] {emotion_state}")

        # ── [8] KV Cache Optimization Strategy ────────────────────────────
        # Kita pisahkan identitas statis agar prefix cache tetap valid.
        # Identitas statis diletakkan di pesan pertama (index 0).
        static_system = {"role": "system", "content": self.system_identity}

        # Ini mencakup waktu, memori, hasil web, dan emosi.
        dynamic_parts = [f"Waktu sekarang: {timestamp_str}."]
        if memory_ctx:
            dynamic_parts.append(f"\n[Memori]\n{memory_ctx}")
        if web_result:
            dynamic_parts.append(f"\n[Hasil Web Search]\n{web_result}")
        if emotion_guidance:
            dynamic_parts.append(f"\n[Panduan Emosi]\n{emotion_guidance}")
        if thought.get("note"):
            dynamic_parts.append(f"\n[Catatan]\n{thought['note']}")
        dynamic_system = {"role": "system", "content": "\n".join(dynamic_parts)}

        self.conversation_history.append({"role": "user", "content": user_input})
        # STRATEGI BARU: Sisipkan pesan dinamis di AKHIR riwayat, tepat sebelum giliran asisten.
        self.conversation_history.append(dynamic_system)


        # ── [9] Token Budget ──────────────────────────────────────────────
        # Pesan dinamis sekarang menjadi bagian dari riwayat, sehingga budget manager akan menanganinya secara otomatis.
        messages_to_send, token_count = self.budget_manager.build_messages(
            system_identity=static_system,
            memory_messages=[], # Dihapus dari sini
            conversation_history=self.conversation_history,
        )

        print(f"[Token] {token_count}/{self.n_ctx} digunakan.")
        sys.stdout.flush()

        # ── [10] Streaming ────────────────────────────────────────────────
        #
        # TIDAK ada llm.reset() sebelum ini.
        # llama.cpp menangani KV cache secara internal — jika prefix
        # berubah, ia akan otomatis menghitung ulang hanya bagian yang perlu.
        #
        spinner = Spinner()
        spinner.start()

        response_stream = self.llama.create_chat_completion(
            messages=messages_to_send,
            max_tokens=128,
            temperature=0.7,
            top_p=0.85,
            top_k=60,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )

        full_response = ""
        first_chunk = True
        for chunk in response_stream:
            if first_chunk:
                spinner.stop()
                sys.stdout.write("Asta: ")
                sys.stdout.flush()
                first_chunk = False
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text = delta["content"]
                sys.stdout.write(text)
                sys.stdout.flush()
                full_response += text

        if first_chunk:
            spinner.stop()

        sys.stdout.write("\n")
        sys.stdout.flush()

        # --- CLEANUP: Hapus pesan sistem dinamis sementara dari riwayat ---
        self.conversation_history.pop()

        self.conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

    def get_session_text(self) -> str:
        lines = []
        for m in self.conversation_history:
            if m["content"]:
                lines.append(f"{m['role']}: {m['content']}")
        return "\n".join(lines)


# ─── Model Loader ─────────────────────────────────────────────────────────────

def load_model(cfg: dict) -> ChatManager:
    choice = cfg.get("model_choice", "1")
    if choice not in MODELS:
        choice = "1"
    model_cfg = MODELS[choice]

    device = cfg.get("device", "cpu")
    use_lora = cfg.get("use_lora", False)

    lora_path = None
    if use_lora and os.path.exists(LORA_ADAPTER_PATH):
        lora_path = LORA_ADAPTER_PATH
        if choice != "2":
            print("[Warn] LoRA adapter dirancang untuk 8B, otomatis switch ke 8B.")
            choice = "2"
            model_cfg = MODELS["2"]

    print(f"\n[Model] Memuat {model_cfg['name']} ({device.upper()})...")

    for path_key in ("model_path", "tokenizer_path"):
        if not Path(model_cfg[path_key]).exists():
            raise FileNotFoundError(f"Tidak ditemukan: {model_cfg[path_key]}")

    n_gpu_layers = 0
    n_threads = os.cpu_count()
    tokenizer = LlamaHFTokenizer.from_pretrained(model_cfg["tokenizer_path"])

    # n_ctx dan n_batch dari config — gunakan nilai yang sama dengan
    # repo Project yang terbukti cepat: n_ctx=8192, n_batch=1024
    n_ctx = cfg.get("token_budget", {}).get("total_ctx", 8192)
    n_batch = cfg.get("n_batch", 1024)

    with contextlib.redirect_stderr(io.StringIO()):
        llama = llama_cpp.Llama(
            model_path=model_cfg["model_path"],
            tokenizer=tokenizer,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mmap=True,
            use_mlock=True,
            n_ctx=n_ctx,
            verbose=True,
            lora_path=lora_path,
            lora_scale=1.0,
            lora_n_gpu_layers=0,
            log_level=0,
        )

    print(f"[Model] Siap! n_ctx={llama.n_ctx()}, n_batch={n_batch}, n_threads={n_threads}\n")
    return ChatManager(llama=llama, system_identity=SYSTEM_IDENTITY, cfg=cfg)