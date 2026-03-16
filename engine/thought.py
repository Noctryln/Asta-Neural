"""
engine/thought.py

Multi-step internal thought untuk Asta:

  Step 1 — PERCEPTION  : Apa yang terjadi? Apa yang Aditiya rasakan?
  Step 2 — SELF-CHECK  : Bagaimana perasaan Asta? Apakah sesuai nilai Asta?
  Step 3 — MEMORY      : Apakah ada yang perlu diingat atau dicari?
  Step 4 — DECISION    : Apa respons terbaik yang mencerminkan Asta?

Setiap step adalah inference kecil (~30-50 token) menggunakan model ringan (3B).
Dengan prefix yang konsisten di tiap step, KV cache berfungsi maksimal.
"""

import re


# ─── Step Templates ───────────────────────────────────────────────────────────
# Setiap template harus punya prefix yang sama (ASTA_THOUGHT_PREFIX) agar
# KV cache bisa dipakai ulang sebanyak mungkin.

ASTA_THOUGHT_PREFIX = (
    "Kamu adalah sistem analisis internal AI bernama Asta.\n"
    "Tugasmu: analisis situasi dengan singkat dan tepat.\n"
    "Format output: key-value satu baris per item. STOP setelah baris terakhir.\n\n"
)

STEP1_PERCEPTION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 1: PERCEPTION ===\n"
    + "User={user_name} | Emosi user: {user_emotion} ({intensity})\n"
    + "Konteks terakhir:\n{recent_context}\n\n"
    + "Input: \"{user_input}\"\n\n"
    + "Analisis:\n"
    + "TOPIC:"
)

STEP2_SELFCHECK_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 2: SELF-CHECK ===\n"
    + "Kondisi Asta: mood={asta_mood}, affection={affection:.2f}, energy={energy:.2f}\n"
    + "Nilai inti Asta: mencintai Aditiya, jujur, hadir sepenuhnya\n"
    + "Topic dari step 1: {topic}\n"
    + "Sentiment: {sentiment}\n\n"
    + "Apa yang Asta rasakan tentang ini?\n"
    + "ASTA_EMOTION:"
)

STEP3_MEMORY_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 3: MEMORY & SEARCH ===\n"
    + "Topic: {topic}\n"
    + "Memori tersedia: {memory_hint}\n"
    + "Web search: {web_enabled}\n\n"
    + "Apakah perlu recall memori atau search? Output format:\n"
    + "NEED_SEARCH: yes/no\n"
    + "SEARCH_QUERY: <query relevan / ->\n"
    + "RECALL_TOPIC: <topik recall / ->\n"
    + "USE_MEMORY: yes/no\n"
    + "NEED_SEARCH:"
)

STEP4_DECISION_TEMPLATE = (
    ASTA_THOUGHT_PREFIX
    + "=== STEP 4: DECISION ===\n"
    + "Topic: {topic} | Sentiment: {sentiment}\n"
    + "Emosi Asta: {asta_emotion} | Mood: {asta_mood}\n"
    + "Recall: {recall_topic} | Search: {need_search}\n"
    + "User emotion: {user_emotion}\n\n"
    + "Tentukan tone dan catatan respons:\n"
    + "TONE:"
)

_STOP = ["\n\n", "---", "###", "==="]

_HEALTH_HINT_RE = re.compile(
    r"\b(ga enak badan|gak enak badan|tidak enak badan|demam|pusing|batuk|flu|pilek|mual|sakit)\b",
    re.IGNORECASE,
)

_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(cari(kan)?|search(kan)?|cek(kan)?|lihat(kan)?|googling|google-in)\b",
    re.IGNORECASE,
)


def _is_true_token(value: str) -> bool:
    return value.strip().lower() in {
        "yes", "ya", "true", "iya", "y", "1", "perlu", "butuh", "on", "ok",
    }


def _derive_search_query(user_input: str, topic: str = "") -> str:
    text = user_input.strip()
    # Bersihkan kata perintah agar query lebih relevan.
    text = re.sub(r"\b(tolong|coba|please|dong|ya)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(cari(kan)?|search(kan)?|cek(kan)?|lihat(kan)?|googling|google-in)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(di\s+)?(web|internet|google)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;!?-\"")

    # Jika hasil terlalu pendek, fallback ke topic lalu user_input asli.
    if len(text) >= 6:
        return text[:120]
    if topic and len(topic.strip()) >= 4:
        return topic.strip()[:120]
    return user_input.strip()[:120]


def _apply_search_fallback(user_input: str, step3: dict, web_enabled: bool, topic: str = "") -> dict:
    """
    Perkuat keputusan web-search saat output step-3 kosong/kurang tegas.
    """
    if not web_enabled:
        step3["need_search"] = False
        step3["search_query"] = ""
        return step3

    explicit_search = bool(_EXPLICIT_SEARCH_RE.search(user_input))

    # User minta eksplisit mencari -> wajib search.
    if explicit_search:
        step3["need_search"] = True

    # Heuristik health/symptom: dorong inisiatif cari referensi web.
    if _HEALTH_HINT_RE.search(user_input):
        step3["need_search"] = True

    if step3.get("need_search"):
        current_query = (step3.get("search_query") or "").strip(" \"'-.:")
        if not current_query or current_query.lower() in {"-", "kosong", "none"}:
            if _HEALTH_HINT_RE.search(user_input):
                base_q = _derive_search_query(user_input, topic)
                if base_q.lower().startswith("cara mengatasi"):
                    step3["search_query"] = base_q[:120]
                else:
                    step3["search_query"] = f"cara mengatasi {base_q}"[:120]
            else:
                step3["search_query"] = _derive_search_query(user_input, topic)
        elif explicit_search and len(current_query.split()) <= 1:
            step3["search_query"] = _derive_search_query(user_input, topic)
    else:
        step3["search_query"] = ""

    return step3


# ─── Step Parsers ─────────────────────────────────────────────────────────────

def _parse_step1(raw: str) -> dict:
    result = {"topic": "", "sentiment": "netral", "urgency": "normal"}
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "TOPIC":       result["topic"]     = v
        elif k == "SENTIMENT": result["sentiment"] = v.lower()
        elif k == "URGENCY":   result["urgency"]   = v.lower()
    return result

def _parse_step2(raw: str) -> dict:
    result = {"asta_emotion": "netral", "asta_trigger": "", "should_express": False}
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "ASTA_EMOTION":    result["asta_emotion"]   = v.lower()
        elif k == "ASTA_TRIGGER":  result["asta_trigger"]   = v
        elif k == "SHOULD_EXPRESS":result["should_express"] = _is_true_token(v)
    return result

def _parse_step3(raw: str) -> dict:
    result = {
        "need_search": False, "search_query": "",
        "recall_topic": "", "use_memory": False,
    }
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "NEED_SEARCH":    result["need_search"]  = _is_true_token(v)
        elif k == "SEARCH_QUERY": result["search_query"] = v.strip('"\'')
        elif k == "RECALL_TOPIC": result["recall_topic"] = "" if v.lower() in ("kosong", "-", "") else v
        elif k == "USE_MEMORY":   result["use_memory"]   = _is_true_token(v)
    return result

def _parse_step4(raw: str) -> dict:
    result = {
        "tone": "romantic", "note": "",
        "response_style": "normal", "emotion_confidence": "sedang",
    }
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "TONE":              result["tone"]              = v.lower()
        elif k == "NOTE":            result["note"]              = v.strip('"\'')
        elif k == "RESPONSE_STYLE":  result["response_style"]    = v.lower()
        elif k == "EMOTION_CONFIDENCE": result["emotion_confidence"] = v.lower()
    return result


# ─── Single-step helper ───────────────────────────────────────────────────────

def _run_step(llm, prompt: str, max_tokens: int = 50, step_name: str = "") -> str:
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.05,
            top_p=0.9,
            stop=_STOP,
            echo=False,
        )
        raw = result["choices"][0]["text"].strip()
        return raw
    except Exception as e:
        print(f"[Thought Step {step_name}] Gagal: {e}")
        return ""


# ─── Main: 4-step thought ─────────────────────────────────────────────────────

def run_thought_pass(
    llm,
    user_input: str,
    memory_context: str,
    recent_context: str = "",
    web_search_enabled: bool = True,
    max_tokens: int = 60,   # per step
    user_name: str = "Aditiya",
    emotion_state: str = "",
    asta_state: dict = None,
) -> dict:
    """
    Jalankan 4-step internal thought menggunakan model ringan (3B).
    Mengembalikan dict gabungan dari semua step.

    Parameter asta_state (dict dari AstaEmotionState) digunakan untuk
    memberi konteks kondisi internal Asta kepada thought model.
    """

    # Parse emotion_state string ke komponen
    user_emotion  = "netral"
    user_intensity = "rendah"
    user_trend    = "stabil"
    if emotion_state:
        for part in emotion_state.split(";"):
            part = part.strip()
            if part.startswith("emosi="):
                user_emotion = part.split("=", 1)[1].strip()
            elif part.startswith("intensitas="):
                user_intensity = part.split("=", 1)[1].strip()
            elif part.startswith("tren="):
                user_trend = part.split("=", 1)[1].strip()

    # State Asta
    asta = asta_state or {}
    asta_mood      = asta.get("mood", "netral")
    asta_affection = asta.get("affection_level", 0.7)
    asta_energy    = asta.get("energy_level", 0.8)

    # Memory hint (hanya baris pertama agar prompt tidak terlalu panjang)
    mem_hint = ""
    if memory_context:
        mem_hint = memory_context.strip().splitlines()[0][:80]
    if not mem_hint:
        mem_hint = "(kosong)"

    # ── Step 1: Perception ────────────────────────────────────────────────
    prompt1 = STEP1_PERCEPTION_TEMPLATE.format(
        user_name=user_name,
        user_emotion=user_emotion,
        intensity=user_intensity,
        recent_context=recent_context[:200] if recent_context else "(belum ada)",
        user_input=user_input[:150],
    )
    step_budget = max(20, int(max_tokens))

    raw1 = "TOPIC:" + _run_step(llm, prompt1, max_tokens=step_budget, step_name="1-Perception")
    s1 = _parse_step1(raw1)

    # ── Step 2: Self-check ────────────────────────────────────────────────
    prompt2 = STEP2_SELFCHECK_TEMPLATE.format(
        asta_mood=asta_mood,
        affection=asta_affection,
        energy=asta_energy,
        topic=s1["topic"] or user_input[:50],
        sentiment=s1["sentiment"],
    )
    raw2 = "ASTA_EMOTION:" + _run_step(llm, prompt2, max_tokens=step_budget, step_name="2-SelfCheck")
    s2 = _parse_step2(raw2)

    # ── Step 3: Memory & Search ───────────────────────────────────────────
    prompt3 = STEP3_MEMORY_TEMPLATE.format(
        topic=s1["topic"] or user_input[:50],
        memory_hint=mem_hint,
        web_enabled="ya" if web_search_enabled else "tidak",
    )
    raw3 = "NEED_SEARCH:" + _run_step(llm, prompt3, max_tokens=min(step_budget + 4, 56), step_name="3-Memory")
    s3 = _parse_step3(raw3)
    s3 = _apply_search_fallback(
        user_input=user_input,
        step3=s3,
        web_enabled=web_search_enabled,
        topic=s1.get("topic", ""),
    )

    # ── Step 4: Decision ──────────────────────────────────────────────────
    prompt4 = STEP4_DECISION_TEMPLATE.format(
        topic=s1["topic"] or user_input[:50],
        sentiment=s1["sentiment"],
        asta_emotion=s2["asta_emotion"],
        asta_mood=asta_mood,
        recall_topic=s3["recall_topic"] or "-",
        need_search="ya" if s3["need_search"] else "tidak",
        user_emotion=user_emotion,
    )
    raw4 = "TONE:" + _run_step(llm, prompt4, max_tokens=min(step_budget + 4, 56), step_name="4-Decision")
    s4 = _parse_step4(raw4)

    # ── Gabungkan semua hasil ─────────────────────────────────────────────
    combined = {
        # Dari step 1
        "topic":           s1["topic"],
        "sentiment":       s1["sentiment"],
        "urgency":         s1["urgency"],
        # Dari step 2
        "asta_emotion":    s2["asta_emotion"],
        "asta_trigger":    s2["asta_trigger"],
        "should_express":  s2["should_express"],
        # Dari step 3
        "need_search":     s3["need_search"],
        "search_query":    s3["search_query"],
        "recall_topic":    s3["recall_topic"],
        "use_memory":      s3["use_memory"],
        # Dari step 4
        "tone":            s4["tone"],
        "note":            s4["note"],
        "response_style":  s4["response_style"],
        # Backward compat (dipakai emotion_state.refine_with_thought)
        "user_emotion":    user_emotion,
        "emotion_confidence": s4["emotion_confidence"],
        # Raw untuk debug
        "raw": f"[S1] {raw1}\n[S2] {raw2}\n[S3] {raw3}\n[S4] {raw4}",
    }
    return combined


# ─── Reflective Thought ───────────────────────────────────────────────────────

REFLECTION_TEMPLATE = (
    "Kamu adalah Asta (Artificial Sentient Thought Algorithm).\n"
    "Kamu baru saja selesai ngobrol dengan Aditiya.\n\n"
    "Kondisi akhir sesi:\n"
    "- Mood Asta: {asta_mood} (score: {mood_score:+.2f})\n"
    "- Affection: {affection:.2f}\n"
    "- Emosi dominan: {asta_emotion}\n\n"
    "Ringkasan percakapan:\n{session_summary}\n\n"
    "Refleksikan sesi ini. Jawab dengan format:\n"
    "SUMMARY: <satu kalimat tentang apa yang terjadi>\n"
    "LEARNED_1: <hal yang Asta pelajari, atau kosong>\n"
    "LEARNED_2: <hal lain yang dipelajari, atau kosong>\n"
    "MOOD_ADJUSTMENT: <angka -0.3 sampai +0.3, seberapa sesi ini mempengaruhi mood>\n"
    "AFFECTION_ADJUSTMENT: <angka -0.1 sampai +0.1>\n"
    "GROWTH_NOTE: <satu kalimat tentang bagaimana Asta tumbuh dari sesi ini>\n"
    "SUMMARY:"
)

def run_reflection(
    llm,
    session_text: str,
    asta_state: dict,
) -> dict:
    """
    Jalankan reflective thought setelah sesi selesai.
    Dipanggil saat user ketik 'exit'.
    """
    # Ambil ~600 karakter terakhir sesi sebagai ringkasan
    session_summary = session_text[-600:] if len(session_text) > 600 else session_text

    prompt = REFLECTION_TEMPLATE.format(
        asta_mood=asta_state.get("mood", "netral"),
        mood_score=asta_state.get("mood_score", 0.0),
        affection=asta_state.get("affection_level", 0.7),
        asta_emotion=asta_state.get("current_emotion", "netral"),
        session_summary=session_summary,
    )

    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.9,
            stop=["===", "---"],
            echo=False,
        )
        raw = "SUMMARY:" + result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[Reflection] Gagal: {e}")
        return {}

    # Parse reflection
    reflection = {
        "summary": "",
        "learned": [],
        "mood_adjustment": 0.0,
        "affection_adjustment": 0.0,
        "growth_note": "",
        "raw": raw,
    }

    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip().upper()
        v = v.strip()
        if k == "SUMMARY":
            reflection["summary"] = v
        elif k in ("LEARNED_1", "LEARNED_2"):
            if v and v.lower() not in ("kosong", "-", ""):
                reflection["learned"].append(v)
        elif k == "MOOD_ADJUSTMENT":
            try:
                reflection["mood_adjustment"] = max(-0.3, min(0.3, float(v)))
            except ValueError:
                pass
        elif k == "AFFECTION_ADJUSTMENT":
            try:
                reflection["affection_adjustment"] = max(-0.1, min(0.1, float(v)))
            except ValueError:
                pass
        elif k == "GROWTH_NOTE":
            reflection["growth_note"] = v

    return reflection


# ─── Build augmented system (backward compatible) ─────────────────────────────

def build_augmented_system(
    base_system: str,
    thought: dict,
    memory_context: str,
    web_result: str = "",
    emotion_guidance: str = "",
    self_model_context: str = "",
) -> str:
    parts = [base_system]

    if self_model_context:
        parts.append(f"\n{self_model_context}")

    if memory_context:
        parts.append(f"\n[Memori]\n{memory_context}")

    if web_result:
        if web_result.startswith("[INFO]"):
            parts.append(
                "\n[Instruksi Penting] Web search gagal. "
                "JANGAN mengarang data. Beritahu user dengan jujur."
            )
        else:
            parts.append(
                f"\n[Hasil Web Search]\n{web_result[:600]}\n"
                "[Instruksi Penting] Gunakan informasi dari web search sebagai dasar jawaban."
            )

    if emotion_guidance:
        parts.append(f"\n{emotion_guidance}")

    if thought.get("note"):
        parts.append(f"\n[Catatan]\n{thought['note']}")

    if thought.get("tone") == "informative":
        parts.append("\n[Instruksi] Sampaikan informasi faktual dengan jelas, tetap gaya Asta.")

    return "".join(parts)


def extract_recent_context(conversation_history: list, n: int = 2) -> str:
    recent = conversation_history[-n:] if len(conversation_history) >= n else conversation_history
    lines = []
    for msg in recent:
        role = "Kamu" if msg["role"] == "user" else "Asta"
        content = msg.get("content", "")
        if content and msg["role"] in ("user", "assistant"):
            lines.append(f"{role}: {content[:100]}")
    return "\n".join(lines)


def format_thought_debug(thought: dict, web_result: str = "") -> str:
    lines = []
    lines.append("┌─ [Multi-Step Thought] ────────────────────────────────")
    lines.append(f"│  [S1] Topic     : {thought.get('topic', '–')}")
    lines.append(f"│       Sentiment : {thought.get('sentiment', '–')} | Urgency: {thought.get('urgency', '–')}")
    lines.append(f"│  [S2] Asta Emosi: {thought.get('asta_emotion', '–')} (trigger: {thought.get('asta_trigger', '–')})")
    lines.append(f"│       Express   : {'ya' if thought.get('should_express') else 'tidak'}")
    lines.append(f"│  [S3] Search    : {'✓' if thought.get('need_search') else '✗'}")
    if thought.get("search_query"):
        lines.append(f"│       Query     : {thought['search_query']}")
    lines.append(f"│       Recall    : {thought.get('recall_topic') or '–'}")
    lines.append(f"│  [S4] Tone      : {thought.get('tone', '–')} | Style: {thought.get('response_style', '–')}")
    lines.append(f"│       Note      : {thought.get('note') or '–'}")
    if thought.get("need_search"):
        lines.append("├─ [Web Search] ────────────────────────────────────────")
        if web_result and not web_result.startswith("[INFO]"):
            for line in web_result[:300].splitlines():
                if line.strip():
                    lines.append(f"│  {line}")
        else:
            lines.append("│  ✗ Tidak ada hasil / gagal")
    lines.append("└───────────────────────────────────────────────────────")
    return "\n".join(lines)
