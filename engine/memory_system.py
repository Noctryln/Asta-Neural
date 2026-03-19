import json
import re
import threading
from pathlib import Path
import numpy as np
import datetime
import torch
from transformers import AutoTokenizer, AutoModel

# ─── Embedding Model ───────────────────────────────────────────────────────────

HF_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_PATH = Path("model") / "embedding_model" / HF_MODEL_NAME.split("/")[-1]

def _load_embedding_model():
    if LOCAL_MODEL_PATH.exists():
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        mdl = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
    else:
        tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        mdl = AutoModel.from_pretrained(HF_MODEL_NAME)
        LOCAL_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(LOCAL_MODEL_PATH)
        mdl.save_pretrained(LOCAL_MODEL_PATH)
    return tok, mdl

_tokenizer, _model = _load_embedding_model()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
        mask_expanded.sum(1), min=1e-9
    )


def create_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(_model.config.hidden_size)
    encoded = _tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = _model(**encoded)
    emb = mean_pooling(out, encoded["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()


def _is_zero_embedding(embedding: list) -> bool:
    """FIX #2: Cek apakah embedding semua nol — sesi kosong/invalid."""
    if not embedding:
        return True
    return np.allclose(np.array(embedding[:10]), 0.0)  # cek 10 elemen pertama saja


# ─── Key Facts Extractor (Fix #5: Diperketat) ─────────────────────────────────

_FACT_PATTERNS = [
    (r"\baku\s+suka\s+([a-zA-Z ]{4,40})", "preferensi"),
    (r"\baku\s+gak\s+suka\s+([a-zA-Z ]{4,40})", "preferensi_tidak"),
    (r"\b(mau|pengen)\s+(ke|pergi|nikah|menikah|liburan)\s+\w+.{0,30}", "rencana"),
    (r"\b(besok|minggu depan|nanti)\s+(kita|aku)\s+\w+.{0,40}", "rencana"),
    (r"\bkita\s+(ke|di)\s+(jepang|bali|jakarta|bandung|surabaya|pantai|gunung)\b.{0,25}", "lokasi"),
    (r"\baku\s+(tinggal|kerja|kuliah)\s+di\s+\w+", "identitas"),
]

_NOISE_RE = re.compile(r"^\s*\*\w+\*\s*$|[*]{2,}")

def extract_key_facts(conversation: list) -> list:
    facts = []
    seen = set()

    for msg in conversation:
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        if not text or len(text) < 15 or _NOISE_RE.search(text):
            continue

        text_lower = text.lower()
        for pattern, category in _FACT_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                fact = match.group(0).strip()
                key = fact[:50]
                if key not in seen and len(fact) >= 12:
                    seen.add(key)
                    facts.append({
                        "category": category,
                        "fact": fact,
                        "source": text[:100],
                    })
                if len(facts) >= 10:
                    return facts
    return facts


def facts_to_text(facts: list) -> str:
    if not facts:
        return ""
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f["category"], []).append(f["fact"])
    lines = []
    for cat, items in by_cat.items():
        lines.append(f"[{cat}] " + "; ".join(items[:2]))
    return "\n".join(lines)


def _build_fallback_summary(conversation: list, key_facts: list, max_chars: int = 240) -> str:
    """Ringkasan minimal saat llm_summary tidak tersedia."""
    if key_facts:
        raw = "; ".join(f.get("fact", "").strip() for f in key_facts if f.get("fact"))
        raw = re.sub(r"\s+", " ", raw).strip(" ;")
        if raw:
            return raw[:max_chars]

    user_msgs = [
        m.get("content", "").strip()
        for m in conversation
        if m.get("role") == "user" and m.get("content")
    ]
    if not user_msgs:
        return ""

    candidate = user_msgs[-1] if len(user_msgs[-1]) >= 12 else user_msgs[0]
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate[:max_chars]


# ─── Base Memory ───────────────────────────────────────────────────────────────

class BaseMemory:
    def __init__(self, file_path: Path, default_content):
        self.file_path = file_path
        self._default = default_content
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            self._write(self._default)
            return self._default
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"[Memory] File {self.file_path.name} rusak, reset.")
            self._write(self._default)
            return self._default

    def _write(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self):
        with self._lock:
            self._write(self.data)

    def save_async(self):
        t = threading.Thread(target=self.save, daemon=True)
        t.start()
        return t


# ─── Semantic Memory ───────────────────────────────────────────────────────────

class SemanticMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "semantic.json", default_content={})
        self._normalize_schema()

    def _normalize_schema(self):
        changed = False
        if not isinstance(self.data, dict):
            self.data = {"facts": {}, "entries": []}
            changed = True
        else:
            if "facts" not in self.data or not isinstance(self.data.get("facts"), dict):
                legacy = {k: v for k, v in self.data.items() if not isinstance(v, list)}
                self.data = {"facts": legacy, "entries": self.data.get("entries", [])}
                changed = True
            if "entries" not in self.data or not isinstance(self.data.get("entries"), list):
                self.data["entries"] = []
                changed = True
        if changed:
            self.save()

    def add_fact(self, key: str, value):
        facts = self.data.setdefault("facts", {})
        if facts.get(key) != value:
            facts[key] = value
            self.save_async()

    def get_fact(self, key: str):
        return self.data.get("facts", {}).get(key)

    def get_all_facts(self) -> dict:
        return self.data.get("facts", {}).copy()

    def remember_web_result(self, query: str, summary: str):
        query = (query or "").strip()
        summary = (summary or "").strip()
        if not query or not summary:
            return
        entries = self.data.setdefault("entries", [])
        compact_summary = re.sub(r"\s+", " ", summary)[:320]
        emb = create_embedding(f"{query}\n{compact_summary}").tolist()
        entry = {
            "kind": "web_search",
            "query": query,
            "summary": compact_summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "embedding": emb,
        }
        replaced = False
        for idx in range(len(entries) - 1, -1, -1):
            existing = entries[idx]
            if existing.get("kind") == "web_search" and existing.get("query", "").lower() == query.lower():
                entries[idx] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        if len(entries) > 80:
            self.data["entries"] = entries[-80:]
        self.save_async()

    def search(self, query: str, top_k: int = 2, threshold: float = 0.18) -> list:
        entries = self.data.get("entries", [])
        if not query or not entries:
            return []
        q_emb = create_embedding(query)
        scored = []
        query_terms = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2}
        for entry in entries:
            emb = entry.get("embedding", [])
            if _is_zero_embedding(emb):
                continue
            sim = float(np.dot(q_emb, np.array(emb)))
            text_blob = f"{entry.get('query', '')} {entry.get('summary', '')}".lower()
            overlap = sum(1 for term in query_terms if term in text_blob)
            score = sim + min(0.24, overlap * 0.04)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]


def _tokenize_topic(text: str) -> list:
    return [w for w in re.split(r"\W+", (text or "").lower()) if len(w) > 2]


def _clip_text(text: str, max_chars: int = 160) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _keyword_overlap_score(text: str, keywords: list, full_query: str = "") -> int:
    hay = (text or "").lower()
    score = 0
    for kw in keywords:
        if kw in hay:
            score += 2
    if full_query and full_query.lower() in hay:
        score += 3
    return score



# ─── Episodic Memory ──────────────────────────────────────────────────────────

class EpisodicMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(directory / "episodic.json", default_content=[])

    def add(self, conversation: list, llm_summary: str = ""):
        text_conv = " ".join(
            f"{m['role']}: {m['content']}"
            for m in conversation
            if m["role"] in ("user", "assistant") and m["content"]
        )
        if not text_conv.strip():
            print("[Episodic] Sesi kosong, tidak disimpan.")
            return

        embedding = create_embedding(text_conv).tolist()
        key_facts = extract_key_facts(conversation)
        final_summary = (llm_summary or "").strip() or _build_fallback_summary(conversation, key_facts)

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "key_facts": key_facts,
            "llm_summary": final_summary,
            "embedding": embedding,
            "conversation": [
                m for m in conversation
                if m["role"] in ("user", "assistant") and m["content"]
            ],
        }
        self.data.append(entry)
        if len(self.data) > 50:
            self.data = self.data[-50:]
        self.save_async()
        print(f"[Episodic] Sesi disimpan. {len(key_facts)} key facts diekstrak.")

    def search(self, query: str, top_k: int = 3, threshold: float = 0.10) -> list:
        if not self.data:
            return []

        q_emb = create_embedding(query)
        sims = []
        valid = []

        for mem in self.data:
            emb = mem.get("embedding", [])
            if _is_zero_embedding(emb):  # FIX #2
                continue
            sim = float(np.dot(q_emb, np.array(emb)))
            sims.append(sim)
            valid.append(mem)

        if not sims:
            return []

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [valid[i] for i in top_idx if sims[i] > threshold]

        if results:
            print(f"[Episodic] Search '{query[:40]}': {len(results)} hasil")
        return results

    def search_by_facts(self, topic: str, top_k: int = 2) -> list:
        if not self.data or not topic:
            return []

        keywords = _tokenize_topic(topic)
        scored = []
        q_emb = create_embedding(topic)

        for entry in self.data:
            emb = entry.get("embedding", [])
            if _is_zero_embedding(emb):
                continue

            lexical_score = 0
            lexical_score += _keyword_overlap_score(entry.get("llm_summary", ""), keywords, topic) * 2
            for kf in entry.get("key_facts", []):
                lexical_score += _keyword_overlap_score(kf.get("fact", ""), keywords, topic) * 3
                lexical_score += _keyword_overlap_score(kf.get("source", ""), keywords)

            for msg in entry.get("conversation", []):
                lexical_score += _keyword_overlap_score(msg.get("content", ""), keywords)

            semantic_score = float(np.dot(q_emb, np.array(emb)))
            total_score = lexical_score + max(0.0, semantic_score) * 4.0
            if lexical_score > 0 or semantic_score >= 0.22:
                scored.append((total_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]

        if results:
            print(f"[Episodic] Fact search '{topic[:40]}': {len(results)} hasil (skor: {[round(s, 2) for s,_ in scored[:top_k]]})")
        return results

    def build_recall_snippets(self, topic: str, top_k: int = 2, max_lines: int = 6) -> list:
        if not topic:
            return []
        entries = self.search_by_facts(topic, top_k=top_k)
        keywords = _tokenize_topic(topic)
        snippets = []
        for entry in entries:
            conv = entry.get("conversation", [])
            lines = []
            matched = False
            for i, msg in enumerate(conv):
                content = msg.get("content", "")
                if not content:
                    continue
                score = _keyword_overlap_score(content, keywords, topic)
                if msg.get("role") == "user" and score > 0:
                    matched = True
                    lines.append(f"User: {_clip_text(content, 120)}")
                    if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant":
                        lines.append(f"Asta: {_clip_text(conv[i + 1].get('content', ''), 120)}")
                elif matched and msg.get("role") == "assistant" and len(lines) < max_lines:
                    lines.append(f"Asta: {_clip_text(content, 120)}")
                if len(lines) >= max_lines:
                    break

            if not lines and entry.get("llm_summary"):
                lines.append(f"Ringkasan: {_clip_text(entry['llm_summary'], 150)}")
            if entry.get("key_facts"):
                facts = [kf.get("fact", "") for kf in entry.get("key_facts", []) if kf.get("fact")]
                if facts:
                    lines.append("Fakta: " + "; ".join(_clip_text(f, 60) for f in facts[:2]))

            if lines:
                snippets.append({
                    "timestamp": entry.get("timestamp", ""),
                    "summary": _clip_text(entry.get("llm_summary", ""), 150),
                    "lines": lines[:max_lines],
                })
        return snippets

    def get_last_n(self, n: int = 3) -> list:
        valid = [s for s in self.data if not _is_zero_embedding(s.get("embedding", []))]
        return valid[-n:]

    def get_recent_facts_text(self, n_sessions: int = 3, max_facts: int = 8) -> str:
        sessions = self.get_last_n(n_sessions)
        all_facts = []
        for s in sessions:
            for f in s.get("key_facts", []):
                # FIX #5: Filter fakta terlalu pendek atau kategori emosi
                if len(f.get("fact", "")) >= 12 and f.get("category") not in ("emosi",):
                    all_facts.append(f)

        priority = {"preferensi": 0, "rencana": 1, "lokasi": 2, "identitas": 3}
        all_facts.sort(key=lambda f: priority.get(f.get("category", ""), 5))
        return facts_to_text(all_facts[:max_facts])


# ─── Core Memory (Fix #3: Profil Pengguna) ────────────────────────────────────

class CoreMemory(BaseMemory):
    def __init__(self, directory: Path):
        super().__init__(
            directory / "core_memory.json",
            default_content={"summary": "", "user_profile": {}}
        )
        if "user_profile" not in self.data:
            self.data["user_profile"] = {}
            self.save_async()

    def get_summary(self) -> str:
        return self.data.get("summary", "")

    def get_profile(self) -> dict:
        return self.data.get("user_profile", {})

    def update_summary(self, text: str, async_save: bool = True):
        if self.data.get("summary") != text:
            self.data["summary"] = text
            if async_save:
                self.save_async()
            else:
                self.save()
            print("[Core Memory] Summary diperbarui.")

    def add_preference(self, preference: str):
        profile = self.data.setdefault("user_profile", {})
        prefs = profile.setdefault("preferensi", [])
        if preference not in prefs:
            prefs.append(preference)
            profile["preferensi"] = prefs[-20:]
            self.save_async()
            print(f"[Core Memory] Preferensi: {preference}")

    def get_context_text(self) -> str:
        parts = []

        summary = self.get_summary()
        if summary:
            clean = re.sub(r'\(Keterangan[^)]*\)', '', summary)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean:
                parts.append(clean[:300])

        profile = self.get_profile()
        if profile:
            lines = []
            if profile.get("preferensi"):
                lines.append("Suka: " + ", ".join(profile["preferensi"][:5]))
            if profile.get("rencana"):
                r = profile["rencana"]
                lines.append("Rencana: " + (", ".join(r[:3]) if isinstance(r, list) else str(r)))
            if lines:
                parts.append("[Profil Pengguna]\n" + "\n".join(lines))

        return "\n\n".join(parts)


# ─── Hybrid Memory ────────────────────────────────────────────────────────────

class HybridMemory:
    def __init__(self, episodic: EpisodicMemory, core: CoreMemory, semantic: SemanticMemory = None):
        self.episodic = episodic
        self.core = core
        self.semantic = semantic

    def build_recall_context(self, topic: str = "", current_query: str = "", max_chars: int = 560) -> str:
        focus = (topic or current_query or "").strip()
        if not focus:
            return ""
        packets = []
        for idx, item in enumerate(self.episodic.build_recall_snippets(focus, top_k=2), start=1):
            header = f"[Recall {idx}]"
            if item.get("summary"):
                header += f" {item['summary']}"
            packets.append(header + "\n" + "\n".join(item.get("lines", [])))
        text = "\n\n".join(packets)
        return _clip_text(text, max_chars) if text else ""

    def get_lightweight_hint(self, current_query: str = "") -> str:
        """Versi ringan untuk model thought agar tidak bloat."""

        parts = []

        # 1. Profil (ringkas)
        if summary := self.core.get_summary():
            parts.append(f"[Profil] {summary[:200]}")

        # 2. Fakta terbaru (1 sesi, max 3 fakta)
        if facts := self.episodic.get_recent_facts_text(n_sessions=1, max_facts=3):
            parts.append(f"[Fakta Baru] {facts}")

        return "\n".join(parts) or "(kosong)"

    def get_context(
        self,
        current_query: str = "",
        recall_topic: str = "",
        max_chars: int = 1200,
        include_recall: bool = True,
    ) -> str:
        parts = []

        memory_intent = bool(re.search(
            r"\b(ingat|ingetin|ingatan|inget|flag\s*point\w*|kemarin|dulu|tadi|apa\s+tadi|apa\s+yang\s+aku\s+bilang|"
            r"kamu\s+ingat|siapa\s+namaku|nama\s+aku|kesukaan\s+aku|hobiku|hobi\s+aku)\b",
            (current_query or ""),
            re.IGNORECASE,
        ))

        core_text = self.core.get_context_text()
        if core_text:
            parts.append(f"[Memori Inti]\n{core_text}")

        facts_text = self.episodic.get_recent_facts_text(n_sessions=3, max_facts=5)
        if facts_text:
            parts.append(f"[Fakta Penting]\n{facts_text}")

        focus = recall_topic if recall_topic and recall_topic.strip().lower() not in ("", "kosong", "-") else ""
        if not focus and memory_intent:
            focus = current_query

        # RECALL: hanya aktif jika dipicu (include_recall)
        if include_recall:
            focus = ""

            # Prioritas 1: recall_topic valid
            if recall_topic and recall_topic.strip().lower() not in {"", "kosong", "-"}:
                focus = recall_topic.strip()

            # Prioritas 2: intent kuat → pakai current_query
            elif memory_intent:
                focus = current_query

            if focus:
                recall_text = self.build_recall_context(
                    topic=focus,
                    current_query=current_query,
                    max_chars=max(220, max_chars // 2),
                )

                if recall_text:
                    parts.append(recall_text)

        if self.semantic and current_query:
            semantic_hits = self.semantic.search(current_query, top_k=1)
            if semantic_hits:
                hit = semantic_hits[0]
                parts.append(
                    f"[Memori Web] Query lama: {_clip_text(hit.get('query', ''), 80)}\n"
                    f"{_clip_text(hit.get('summary', ''), 220)}"
                )

        full_text = "\n\n".join(part for part in parts if part)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."
        return full_text

    def extract_and_save_preferences(self, conversation: list):
        pref_re = re.compile(r"\baku\s+suka\s+([a-zA-Z ]{4,30})", re.IGNORECASE)
        for msg in conversation:
            if msg.get("role") != "user":
                continue
            for match in pref_re.finditer(msg.get("content", "")):
                pref = match.group(1).strip().lower()
                if len(pref) >= 4 and pref not in ("kamu", "asta", "sama", "banget", "juga"):
                    self.core.add_preference(pref)

    def update_core_async(self, llm_callable, current_session_text: str):
        def _worker():
            old_summary = self.core.get_summary()
            combined = ""
            if old_summary:
                clean = re.sub(r'\(Keterangan[^)]*\)', '', old_summary).strip()
                combined += f"Ringkasan sebelumnya:\n{clean[:400]}\n\n"
            combined += f"Percakapan terbaru:\n{current_session_text[:800]}"

            prompt = (
                "Berdasarkan ringkasan sebelumnya dan percakapan terbaru, "
                "buat satu paragraf ringkas (maks 100 kata) tentang fakta penting pengguna. "
                "Fokus: nama, preferensi, rencana konkret, hubungan dengan Asta. "
                "Bahasa Indonesia. JANGAN tambahkan keterangan atau catatan.\n\n"
                f"{combined}\n\nRingkasan:"
            )
            try:
                result = llm_callable(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.1,
                    stop=["\n\n", "###", "(Keterangan"],
                )
                summary = result["choices"][0]["text"].strip()
                if summary:
                    self.core.update_summary(summary, async_save=True)
                    print("[Core Memory] Background update selesai.")
            except Exception as e:
                print(f"[Core Memory] Background update gagal: {e}")

        thread = threading.Thread(target=_worker, daemon=False)
        thread.start()
        return thread
