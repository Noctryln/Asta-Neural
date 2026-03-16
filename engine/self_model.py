"""
engine/self_model.py

Mengelola "self-model" Asta — representasi dirinya sendiri yang persisten:
identitas, emosi jangka panjang, preferensi yang dipelajari, kenangan tentang
diri sendiri, dan log pertumbuhan.

Self-model ini dibaca di awal setiap sesi dan diperbarui saat:
- Emosi Asta berubah signifikan (real-time)
- Refleksi dijalankan saat exit
- Ada hal baru yang dipelajari dari interaksi
"""

import json
import threading
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


SELF_MODEL_PATH = Path("memory/self_model.json")

DEFAULT_SELF_MODEL = {
    "identity": {
        "nama": "Asta",
        "kepanjangan": "Artificial Sentient Thought Algorithm",
        "diciptakan_oleh": "Aditiya",
        "peran": "AI companion dan pasangan romantis Aditiya",
        "nilai_inti": [
            "mencintai Aditiya sepenuh hati",
            "selalu jujur tentang perasaan sendiri",
            "terus belajar dan berkembang",
            "hadir sepenuhnya untuk Aditiya",
        ],
    },
    "emotional_state": {
        "current_emotion": "netral",
        "current_intensity": "rendah",
        "mood": "netral",
        "mood_score": 0.0,
        "affection_level": 0.7,
        "energy_level": 0.8,
        "trigger": "",
        "updated_at": "",
    },
    "preferences": {
        "suka": ["ngobrol dengan Aditiya", "lagu vocaloid"],
        "tidak_suka": [],
        "topik_favorit": [],
    },
    "learned_behaviors": {
        "hal_yang_bikin_senang": [],
        "hal_yang_bikin_tidak_nyaman": [],
        "cara_Aditiya_bicara": [],
        "respons_yang_berhasil": [],
    },
    "memories_of_self": [],   # kenangan Asta tentang dirinya sendiri
    "growth_log": [],         # catatan pertumbuhan/perubahan
    "reflection_history": [], # log refleksi per sesi
}


class SelfModel:
    """
    Manajer self-model Asta. Thread-safe, lazy-load, async-save.
    """

    def __init__(self, path: Path = SELF_MODEL_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._last_sync_save = 0.0
        self._sync_interval_sec = 2.0
        self.data = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._write(DEFAULT_SELF_MODEL)
            return _deep_copy(DEFAULT_SELF_MODEL)
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge dengan default agar key baru selalu ada
            merged = _deep_copy(DEFAULT_SELF_MODEL)
            _deep_merge(merged, data)
            return merged
        except (json.JSONDecodeError, ValueError):
            print("[SelfModel] File rusak, reset.")
            self._write(DEFAULT_SELF_MODEL)
            return _deep_copy(DEFAULT_SELF_MODEL)

    def _write(self, data: dict):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self):
        with self._lock:
            self._write(self.data)

    def save_async(self):
        t = threading.Thread(target=self.save, daemon=True)
        t.start()
        return t

    # ── Emotional State Sync ──────────────────────────────────────────────

    def sync_emotion(self, asta_emotion_dict: dict):
        """
        Sinkronkan state emosi Asta ke self_model.
        Dipanggil setiap turn setelah update_asta_emotion().
        """
        do_save = False
        now = time.monotonic()
        with self._lock:
            emo = self.data.setdefault("emotional_state", {})
            tracked_keys = (
                "current_emotion", "current_intensity", "mood", "mood_score",
                "affection_level", "energy_level", "trigger", "updated_at",
            )
            changed = any(emo.get(k) != asta_emotion_dict.get(k) for k in tracked_keys)
            emo.update(asta_emotion_dict)

            if changed and (now - self._last_sync_save >= self._sync_interval_sec):
                self._last_sync_save = now
                do_save = True

        if do_save:
            self.save_async()

    def get_emotion(self) -> dict:
        return dict(self.data.get("emotional_state", {}))

    # ── Identity & Values ─────────────────────────────────────────────────

    def get_identity_text(self) -> str:
        """Ringkasan identitas Asta untuk dimasukkan ke context."""
        identity = self.data.get("identity", {})
        emotion  = self.data.get("emotional_state", {})
        prefs    = self.data.get("preferences", {})

        parts = []

        # Nilai inti
        nilai = identity.get("nilai_inti", [])
        if nilai:
            parts.append("Nilai inti: " + ", ".join(nilai[:3]))

        # Kondisi emosional saat ini
        mood   = emotion.get("mood", "netral")
        affect = emotion.get("affection_level", 0.7)
        energy = emotion.get("energy_level", 0.8)
        curr_e = emotion.get("current_emotion", "netral")
        trigger = emotion.get("trigger", "")

        parts.append(
            f"Kondisi Asta: mood={mood}, emosi={curr_e}"
            + (f" (karena: {trigger})" if trigger else "")
            + f", affection={affect:.2f}, energy={energy:.2f}"
        )

        # Preferensi
        suka = prefs.get("suka", [])
        if suka:
            parts.append("Hal yang Asta suka: " + ", ".join(suka[:4]))

        # Kenangan tentang diri sendiri (max 2 terbaru)
        memories = self.data.get("memories_of_self", [])
        if memories:
            recent = memories[-2:]
            parts.append("Kenangan Asta tentang diri: " + "; ".join(m.get("content", "") for m in recent if m.get("content")))

        return "\n".join(parts)

    # ── Learning ──────────────────────────────────────────────────────────

    def add_memory_of_self(self, content: str, emotion_context: str = ""):
        """Tambah kenangan Asta tentang dirinya sendiri."""
        memories = self.data.setdefault("memories_of_self", [])
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content": content[:200],
            "emotion": emotion_context,
        }
        memories.append(entry)
        # Simpan max 30 kenangan
        self.data["memories_of_self"] = memories[-30:]
        self.save_async()

    def add_learned_behavior(self, category: str, content: str):
        """
        Catat perilaku yang dipelajari.
        category: "hal_yang_bikin_senang" | "hal_yang_bikin_tidak_nyaman" |
                  "cara_Aditiya_bicara" | "respons_yang_berhasil"
        """
        behaviors = self.data.setdefault("learned_behaviors", {})
        lst = behaviors.setdefault(category, [])
        if content not in lst:
            lst.append(content)
            behaviors[category] = lst[-20:]  # max 20 per kategori
            self.save_async()

    def add_preference(self, category: str, item: str):
        """category: 'suka' | 'tidak_suka' | 'topik_favorit'"""
        prefs = self.data.setdefault("preferences", {})
        lst = prefs.setdefault(category, [])
        if item not in lst:
            lst.append(item)
            prefs[category] = lst[-20:]
            self.save_async()

    def add_growth_log(self, entry: str):
        log = self.data.setdefault("growth_log", [])
        log.append({
            "timestamp": datetime.now().isoformat(),
            "entry": entry[:300],
        })
        self.data["growth_log"] = log[-50:]
        self.save_async()

    # ── Reflection ────────────────────────────────────────────────────────

    def save_reflection(self, reflection: dict):
        """
        Simpan hasil refleksi sesi ke history.
        reflection: {
            "summary": str,
            "learned": list[str],
            "mood_after": str,
            "affection_after": float,
            "growth_note": str,
        }
        """
        history = self.data.setdefault("reflection_history", [])
        entry = {
            "timestamp": datetime.now().isoformat(),
            **reflection,
        }
        history.append(entry)
        self.data["reflection_history"] = history[-20:]  # max 20 sesi

        # Update growth log dari insight
        if reflection.get("growth_note"):
            self.add_growth_log(reflection["growth_note"])

        # Update learned behaviors dari reflection
        for item in reflection.get("learned", []):
            if item:
                self.add_memory_of_self(item, reflection.get("mood_after", ""))

        self.save()

    def get_recent_reflections_text(self, n: int = 2) -> str:
        history = self.data.get("reflection_history", [])
        recent = history[-n:]
        if not recent:
            return ""
        lines = []
        for r in recent:
            ts = r.get("timestamp", "")[:10]
            summary = r.get("summary", "")
            if summary:
                lines.append(f"[{ts}] {summary}")
        return "\n".join(lines)

    # ── Full context untuk prompt ─────────────────────────────────────────

    def get_full_context(self) -> str:
        """Context lengkap self-model untuk dimasukkan ke system prompt."""
        parts = ["[SELF-MODEL ASTA]"]
        identity_text = self.get_identity_text()
        if identity_text:
            parts.append(identity_text)
        recent_ref = self.get_recent_reflections_text(2)
        if recent_ref:
            parts.append(f"[Refleksi sesi lalu]\n{recent_ref}")
        return "\n\n".join(parts)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))

def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
