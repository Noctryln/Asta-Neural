import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from engine.thought import run_thought_pass


@dataclass
class TestCase:
    user_input: str
    expected: str  # search | recall | casual


class MockThoughtLLM:
    """Fallback LLM untuk uji parser/thought pipeline tanpa model lokal."""

    def create_completion(self, prompt, max_tokens=60, temperature=0.05, top_p=0.9, stop=None, echo=False):
        return {"choices": [{"text": self._respond(prompt)}]}

    @staticmethod
    def _extract_user_input(prompt: str) -> str:
        for marker in ("Input User: \"", "Input: \""):
            if marker in prompt:
                return prompt.split(marker, 1)[1].split("\"", 1)[0].strip()
        return ""

    def _respond(self, prompt: str) -> str:
        p = prompt.lower()
        user_input = self._extract_user_input(prompt).lower()

        if "=== step 1" in p:
            if "flag point" in user_input:
                return "TOPIC: menanyakan flag point\nSENTIMENT: netral\nURGENCY: normal"
            if "film horor" in user_input:
                return "TOPIC: rekomendasi film horor bioskop minggu ini\nSENTIMENT: netral\nURGENCY: normal"
            if "layarnya biru" in user_input:
                return "TOPIC: solusi laptop blue screen\nSENTIMENT: negatif\nURGENCY: normal"
            return "TOPIC: percakapan santai\nSENTIMENT: positif\nURGENCY: normal"

        if "=== step 2" in p:
            return "ASTA_EMOTION: netral\nASTA_TRIGGER: memahami kebutuhan user\nSHOULD_EXPRESS: no"

        if "=== step 3" in p:
            if "film horor" in user_input:
                return (
                    "REASONING: Butuh info terbaru bioskop.\n"
                    "NEED_SEARCH: yes\n"
                    "SEARCH_QUERY: film horor yang tayang di bioskop minggu ini terbaik\n"
                    "RECALL_TOPIC: -\n"
                    "USE_MEMORY: no"
                )
            if "flag point" in user_input:
                return (
                    "REASONING: Ini topik memori pribadi, tidak perlu web.\n"
                    "NEED_SEARCH: no\n"
                    "SEARCH_QUERY: -\n"
                    "RECALL_TOPIC: flag point rahasia aditiya dan asta\n"
                    "USE_MEMORY: yes"
                )
            if "layarnya biru" in user_input:
                return (
                    "REASONING: Masalah teknis butuh referensi solusi terkini. NEED_SEARCH: yes\n"
                    "SEARCH_QUERY: cara memperbaiki laptop blue screen\n"
                    "RECALL_TOPIC: -\n"
                    "USE_MEMORY: no"
                )
            return (
                "REASONING: Hanya obrolan santai/ucapan terima kasih.\n"
                "NEED_SEARCH: no\n"
                "SEARCH_QUERY: -\n"
                "RECALL_TOPIC: -\n"
                "USE_MEMORY: no"
            )

        if "=== step 4" in p:
            return (
                "TONE: hangat\n"
                "NOTE: Jawab singkat sesuai kebutuhan user.\n"
                "RESPONSE_STYLE: normal\n"
                "USER_EMOTION: netral\n"
                "EMOTION_CONFIDENCE: sedang"
            )
        return ""


def load_real_llm():
    from engine.model import MODELS, _load_llama

    model_info = MODELS["1"]  # paksa Qwen2.5 3B
    model_path = Path(model_info["model_path"])
    tokenizer_path = Path(model_info["tokenizer_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model Qwen2.5 3B tidak ditemukan: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer Qwen2.5 3B tidak ditemukan: {tokenizer_path}")

    print(f"Loading real model: {model_path}")
    return _load_llama(str(model_path), str(tokenizer_path), n_ctx=2048, n_batch=512)


def _assert_case(thought: dict, expected: str) -> bool:
    if expected == "search":
        return (
            thought["need_search"] is True
            and bool(thought["search_query"].strip())
            and thought["recall_topic"] == ""
            and thought["use_memory"] is False
        )
    if expected == "recall":
        return (
            thought["need_search"] is False
            and thought["search_query"] == ""
            and bool(thought["recall_topic"].strip())
            and thought["use_memory"] is True
        )
    return (
        thought["need_search"] is False
        and thought["search_query"] == ""
        and thought["recall_topic"] == ""
        and thought["use_memory"] is False
    )


def run_rigorous_debug(llm, loops: int = 8) -> bool:
    user_name = "Aditiya"
    memory_hint = "[Memori Inti]\nAditiya dan Asta punya 'flag point' rahasia."
    asta_state = {"mood": "senang", "affection_level": 0.85, "energy_level": 0.9}

    test_cases = [
        TestCase("Film horor yang lagi tayang di bioskop minggu ini yang bagus apa?", "search"),
        TestCase("Wahh kayaknya serem ya filmnya...", "casual"),
        TestCase("Inget gak flag point kita apa?", "recall"),
        TestCase("Gimana cara benerin laptop yang layarnya biru?", "search"),
        TestCase("Hehe makasih ya infonya sayang", "casual"),
    ]

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    config["disable_step3_rule_based"] = True

    for epoch in range(1, loops + 1):
        print(f"\n===== DEBUG LOOP #{epoch} =====")
        all_pass = True
        recent_context = "Asta: Halo sayang!\nAditiya: Halo Asta!"

        for i, tc in enumerate(test_cases, 1):
            thought = run_thought_pass(
                llm=llm,
                user_input=tc.user_input,
                memory_context=memory_hint,
                recent_context=recent_context,
                web_search_enabled=True,
                user_name=user_name,
                asta_state=asta_state,
                cfg=config,
            )

            ok = _assert_case(thought, tc.expected)
            all_pass = all_pass and ok
            status = "PASS" if ok else "FAIL"

            print(f"[{status}] TURN {i} expected={tc.expected}")
            print(
                f"  NEED_SEARCH={thought['need_search']} | "
                f"SEARCH_QUERY={thought['search_query'] or '-'} | "
                f"RECALL_TOPIC={thought['recall_topic'] or '-'} | "
                f"USE_MEMORY={thought['use_memory']}"
            )

            recent_context += f"\nAditiya: {tc.user_input}\nAsta: (responded)"
            recent_context = recent_context[-500:]

        if all_pass:
            print("\n✅ BERHASIL: model/pipeline mandiri memutuskan NEED_SEARCH, SEARCH_QUERY, RECALL_TOPIC, USE_MEMORY.")
            return True

    print("\n❌ BELUM BERHASIL dalam batas loop.")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mock", action="store_true", help="Pakai mock LLM (fallback).")
    parser.add_argument("--loops", type=int, default=8)
    args = parser.parse_args()

    if args.use_mock:
        print("[INFO] Menjalankan debug dengan MOCK LLM.")
        ok = run_rigorous_debug(MockThoughtLLM(), loops=args.loops)
        raise SystemExit(0 if ok else 1)

    print("[INFO] Menjalankan debug dengan model langsung: Qwen2.5 3B Instruct.")
    try:
        llm = load_real_llm()
    except Exception as e:
        print(f"[ERROR] Gagal load model real: {e}")
        print("[INFO] Pastikan dependency 'llama_cpp' terpasang dan file model/tokenizer Qwen2.5-3B tersedia di folder ./model.")
        raise SystemExit(1)

    ok = run_rigorous_debug(llm, loops=args.loops)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
