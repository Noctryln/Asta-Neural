from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional
import re
import math


# ─── Emotion & Mood Constants ─────────────────────────────────────────────────

VALID_EMOTIONS = {"netral", "sedih", "cemas", "marah", "senang", "romantis", "rindu", "bangga", "kecewa"}

# Valence: positif = +, negatif = -
EMOTION_VALENCE = {
    "netral":   0.0,
    "senang":   0.8,
    "romantis": 0.9,
    "bangga":   0.7,
    "rindu":    0.2,
    "sedih":   -0.7,
    "kecewa":  -0.5,
    "cemas":   -0.4,
    "marah":   -0.8,
}

MOOD_DECAY_RATE = 0.12
USER_TO_ASTA_INFLUENCE = 0.25
SELF_REINFORCEMENT = 0.35

# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class UserEmotionState:
    """Emosi yang terdeteksi dari input user."""
    user_emotion: str = "netral"
    intensity: str = "rendah"
    trend: str = "stabil"
    turns_in_state: int = 0
    last_user_text: str = ""
    updated_at: str = ""


@dataclass
class AstaEmotionState:
    current_emotion: str = "netral"
    current_intensity: str = "rendah"
    mood: str = "netral"
    mood_score: float = 0.0        # -1.0 (sangat buruk) sampai +1.0 (sangat baik)
    affection_level: float = 0.7   # 0.0 – 1.0, naik turun berdasarkan interaksi
    energy_level: float = 0.8      # 0.0 – 1.0, mempengaruhi keaktifan respons
    trigger: str = ""              # apa yang memicu emosi saat ini
    updated_at: str = ""


@dataclass
class CombinedEmotionState:
    user: UserEmotionState = field(default_factory=UserEmotionState)
    asta: AstaEmotionState = field(default_factory=AstaEmotionState)


# ─── User Emotion Manager ─────────────────────────────────────────────────────

class UserEmotionDetector:
    # ─── 1. Pattern Definitions (Weighted) ────────────────────────────────────
    # Format: "regex_pattern": weight (1=low, 2=medium, 3=high)
    
    _PATTERNS = {
        "sedih": {
            r"\b(sedih|murung|suram)\b": 2,
            r"\b(nangis|menangis|air mata)\b": 3,
            r"\b(hancur|remuk|sakit hati|perih hati)\b": 3,
            r"\b(putus asa|nyerah|lelah hati|capek bgt|capek banget)\b": 3,
            r"\b(kecewa|galau|bad mood|badmood)\b": 2,
            r"\b(sendiri|sepi|kesepian)\b": 1,
        },
        "cemas": {
            r"\b(takut|seram|ngeri|merinding)\b": 2,
            r"\b(cemas|khawatir|gelisah|was-was)\b": 2,
            r"\b(panik|stres|deg-degan|jantung)\b": 3,
            r"\b(overthinking|kepikiran terus)\b": 2,
            # Fisik (Urgent)
            r"\b(sakit|nyeri|perih|luka|darah|pusing|mual|sesak)\b": 3,
            r"\b(tolong|bantu|bahaya|darurat)\b": 2,
        },
        "marah": {
            r"\b(marah|emosi|amuk)\b": 2,
            r"\b(kesal|sebal|jengkel|bete|badmood)\b": 1,
            r"\b(benci|muak|jijik|ilfil)\b": 3,
            r"\b(bangsat|anjing|babi|goblok|tolol|bodoh|dungu|sialan)\b": 3,
            r"\b(nyebelin|parah|keterlaluan)\b": 2,
            r"\b(gak suka|tidak suka)\b": 1,
        },
        "senang": {
            r"\b(senang|happy|bahagia|gembira)\b": 2,
            r"\b(seru|asik|menarik|keren|mantap)\b": 2,
            r"\b(lucu|kocak|ngakak|wkwk|haha|hihi|lol)\b": 2,
            r"\b(semangat|excited|antusias)\b": 2,
            r"\b(makasih|terima kasih|thanks|thank you)\b": 1, # Makasih = Senang ringan
            r"\b(bersyukur|lega|plong)\b": 2,
        },
        "romantis": {
            r"\b(cinta|love|sayang|kasih sayang)\b": 3, # "kasih" sendirian dihapus
            r"\b(kangen|rindu|miss)\b": 3,
            r"\b(peluk|cium|kiss|hug)\b": 2,
            r"\b(manja|mesra|gombal)\b": 2,
            r"\b(cantik|ganteng|cakep|manis)\b": 1, 
            r"\b(jadian|pacar|pasangan|soulmate)\b": 2,
        },
        "bangga": {
            r"\b(bangga|proud|hebat)\b": 3,
            r"\b(berhasil|sukses|lulus|juara)\b": 2,
            r"\b(pencapaian|prestasi)\b": 2,
        },
        "kecewa": {
            r"\b(kecewa|mengecewakan)\b": 3,
            r"\b(sayang banget|sayang sekali)\b": 1, # Konteks: "yah sayang banget tiket habis"
            r"\b(gagal|kalah|rugi)\b": 2,
        }
    }

    _NEGATIONS = r"\b(tidak|gak|enggak|bukan|jangan|jan|ndak)\s+"

    _HOSTILE_TARGET_PATTERNS = [
        r"\b(kamu|lu|elo)\s+(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b",
        r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b.{0,12}\b(kamu|lu|elo|asta)\b",
    ]

    def __init__(self):
        self.state = UserEmotionState(updated_at=datetime.now().isoformat())

    def _score_emotions(self, text: str) -> dict:
        text_lower = text.lower().strip()
        scores = {k: 0 for k in self._PATTERNS}
        
        # 1. Cek Pola Dasar
        for emotion, patterns in self._PATTERNS.items():
            for pattern, weight in patterns.items():
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    # Cek Negasi (Lookbehind manual sederhana)
                    start_idx = match.start()
                    preceding_text = text_lower[max(0, start_idx-10):start_idx]
                    if re.search(self._NEGATIONS, preceding_text):
                        continue # Skip jika ada negasi ("gak sedih")
                    
                    scores[emotion] += weight

        # 2. Logika Khusus "Makasih" -> Netral/Senang (Bukan Romantis)
        # Jika cuma bilang "makasih", jangan biarkan skor romantis naik
        if re.search(r"\b(makasih|terima kasih|thanks)\b", text_lower):
            # Jika tidak ada kata "sayang/cinta", pastikan romantis 0
            if scores["romantis"] > 0:
                is_romantic_explicit = re.search(r"\b(cinta|sayang|love|kangen|rindu)\b", text_lower)
                if not is_romantic_explicit:
                    scores["romantis"] = 0 

        # 3. Logika Tanda Baca
        # Tanda tanya (?) cenderung Netral atau Cemas (jika ada kata tanya)
        if "?" in text:
            pass 
        # Tanda seru (!) meningkatkan skor emosi dominan, bukan membuat senang
        if "!" in text:
            # Cari emosi tertinggi saat ini
            top_emo = max(scores, key=scores.get)
            if scores[top_emo] > 0:
                scores[top_emo] += 1

        return scores

    def _intensity_from_text(self, text: str, score: int) -> str:
        # Deteksi Capslock (jika > 50% huruf besar dan panjang > 5)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        is_caps = caps_ratio > 0.6 and len(text) > 5

        # Kata penguat
        intensifiers = re.search(r"\b(banget|bgt|parah|sangat|bener|sekali)\b", text, re.IGNORECASE)
        
        if score >= 4 or (score >= 2 and (is_caps or intensifiers)):
            return "tinggi"
        if score >= 2:
            return "sedang"
        return "rendah"

    def update(self, user_text: str, recent_context: str = "") -> UserEmotionState:
        # Analisis hanya pada teks terbaru untuk akurasi saat ini
        scores = self._score_emotions(user_text)

        # Hostility Check
        hostility_hits = sum(
            1 for p in self._HOSTILE_TARGET_PATTERNS
            if re.search(p, user_text, re.IGNORECASE)
        )
        if hostility_hits > 0:
            scores["marah"] += 3 * hostility_hits
            scores["kecewa"] += 2 * hostility_hits

        # Tentukan Pemenang
        detected = "netral"
        top_score = 0
        
        for emotion, score in scores.items():
            if score > top_score:
                top_score = score
                detected = emotion
        
        # Ambang batas minimum: Jika skor terlalu kecil (<1), tetap Netral
        # Kecuali 'makasih' (skor 1) bisa masuk ke senang/netral
        if top_score < 1:
            detected = "netral"
        elif top_score == 1 and detected == "senang":
            # "Makasih" (skor 1) seringkali netral sopan
            if len(user_text.split()) < 4: 
                detected = "netral" # "Makasih ya" -> Netral
            else:
                detected = "senang" # "Makasih banget hadiahnya" -> Senang

        # Hitung Tren & Intensitas
        intensity = self._intensity_from_text(user_text, top_score)
        
        prev = self.state.user_emotion
        if detected == prev:
            turns = self.state.turns_in_state + 1
            trend = "stabil"
        else:
            turns = 1
            # Logika trend sederhana
            bad_emos = {"sedih", "cemas", "marah", "kecewa"}
            if prev in bad_emos and detected not in bad_emos:
                trend = "membaik"
            elif prev not in bad_emos and detected in bad_emos:
                trend = "memburuk"
            else:
                trend = "stabil"

        self.state = UserEmotionState(
            user_emotion=detected,
            intensity=intensity,
            trend=trend,
            turns_in_state=turns,
            last_user_text=user_text[:120],
            updated_at=datetime.now().isoformat(),
        )
        return self.state

    def refine_with_thought(self, thought: dict) -> UserEmotionState:
        llm_emotion = (thought.get("user_emotion") or "").strip().lower()
        llm_conf = (thought.get("emotion_confidence") or "").strip().lower()
        if llm_emotion not in VALID_EMOTIONS:
            return self.state
        if llm_conf == "tinggi" or (llm_conf == "sedang" and self.state.user_emotion == "netral"):
            self.state.user_emotion = llm_emotion
            if llm_conf == "tinggi" and llm_emotion in {"sedih", "cemas", "marah"}:
                if self.state.intensity == "rendah":
                    self.state.intensity = "sedang"
            if llm_conf == "tinggi":
                self.state.trend = "memburuk" if llm_emotion in {"sedih", "cemas", "marah"} else self.state.trend
        return self.state

    def get_state(self) -> UserEmotionState:
        return self.state


# ─── Asta Emotion Manager ─────────────────────────────────────────────────────

class AstaEmotionManager:
    def __init__(self, initial_state: Optional[AstaEmotionState] = None):
        self.state = initial_state or AstaEmotionState(
            updated_at=datetime.now().isoformat()
        )

    def _mood_to_label(self, score: float) -> str:
        if score >= 0.6:  return "sangat senang"
        if score >= 0.3:  return "senang"
        if score >= 0.1:  return "sedikit senang"
        if score >= -0.1: return "netral"
        if score >= -0.3: return "sedikit murung"
        if score >= -0.6: return "murung"
        return "sangat murung"

    def _score_to_intensity(self, score: float) -> str:
        abs_score = abs(score)
        if abs_score >= 0.6: return "tinggi"
        if abs_score >= 0.3: return "sedang"
        return "rendah"

    def update_from_interaction(
        self,
        user_emotion: UserEmotionState,
        thought_result: dict,
        user_text: str,
    ) -> AstaEmotionState:

        now = datetime.now().isoformat()

        # ── Ambil keputusan emosi Asta dari thought ────────────────────────
        asta_emotion_from_thought = (thought_result.get("asta_emotion") or "").strip().lower()
        asta_trigger = thought_result.get("asta_trigger", "")

        # ── Decay mood ke arah netral (baseline) ──────────────────────────
        current_score = self.state.mood_score
        current_score *= (1.0 - MOOD_DECAY_RATE)

        # ── Pengaruh emosi user ke mood Asta ──────────────────────────────
        user_valence = EMOTION_VALENCE.get(user_emotion.user_emotion, 0.0)
        user_intensity_mult = {"rendah": 0.5, "sedang": 1.0, "tinggi": 1.5}.get(user_emotion.intensity, 1.0)
        user_influence = user_valence * user_intensity_mult * USER_TO_ASTA_INFLUENCE
        current_score += user_influence

        # ── Pengaruh emosi Asta sendiri (self-reinforcement) ───────────────
        if asta_emotion_from_thought in VALID_EMOTIONS:
            asta_valence = EMOTION_VALENCE.get(asta_emotion_from_thought, 0.0)
            current_score += asta_valence * SELF_REINFORCEMENT

        # ── Clamp score ke [-1.0, 1.0] ────────────────────────────────────
        current_score = max(-1.0, min(1.0, current_score))

        # ── Tentukan emosi dominan Asta ───────────────────────────────────
        if asta_emotion_from_thought in VALID_EMOTIONS:
            dominant_emotion = asta_emotion_from_thought
        elif current_score >= 0.5:
            dominant_emotion = "senang"
        elif current_score >= 0.2:
            dominant_emotion = "senang"
        elif current_score <= -0.5:
            dominant_emotion = "sedih"
        elif current_score <= -0.2:
            dominant_emotion = "cemas"
        else:
            dominant_emotion = "netral"

        # ── Sinyal hostility user ke Asta (untuk kalibrasi lebih realistis) ─
        hostile_to_asta = bool(re.search(
            r"\b(kamu|asta|lu|elo)\b.{0,12}\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b|"
            r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b.{0,12}\b(kamu|asta|lu|elo)\b|"
            r"\baku\s+marah\s+(banget\s+)?(sama|ke)\s+(kamu|asta)\b",
            user_text or "",
            re.IGNORECASE,
        ))
        repeated_insult = bool(
            user_emotion.turns_in_state >= 2
            and re.search(r"\b(bodoh|tolol|goblok|dungu|payah|nyebelin|jelek)\b", user_text or "", re.IGNORECASE)
        )

        if (hostile_to_asta or repeated_insult) and user_emotion.user_emotion in {"marah", "kecewa"}:
            current_score -= 0.18 if user_emotion.intensity == "tinggi" else 0.12

        # ── Update affection berdasarkan interaksi romantis/hostile ───────
        affection = self.state.affection_level
        if user_emotion.user_emotion == "romantis":
            affection = min(1.0, affection + 0.02)
        elif user_emotion.user_emotion in {"marah", "kecewa"}:
            drop = 0.015
            if user_emotion.intensity == "tinggi":
                drop += 0.015
            if user_emotion.turns_in_state >= 2:
                drop += 0.01
            if hostile_to_asta or repeated_insult:
                drop += 0.02
            affection = max(0.1, affection - drop)
        # Affection sangat perlahan decay
        affection = affection * 0.999 + 0.7 * 0.001  # selalu menuju 0.7 perlahan

        # Override emosi dominan jika ada hostility kuat (hindari tetap "senang")
        if (hostile_to_asta or repeated_insult) and user_emotion.user_emotion in {"marah", "kecewa"}:
            if user_emotion.intensity == "tinggi" or user_emotion.turns_in_state >= 2:
                dominant_emotion = "sedih"
            else:
                dominant_emotion = "kecewa"

        # ── Update energy ─────────────────────────────────────────────────
        energy = self.state.energy_level
        if dominant_emotion in {"senang", "romantis", "bangga"}:
            energy = min(1.0, energy + 0.05)
        elif dominant_emotion in {"sedih", "cemas", "marah"}:
            energy = max(0.2, energy - 0.03)
        else:
            # Decay ke 0.8 (baseline)
            energy = energy * 0.95 + 0.8 * 0.05

        self.state = AstaEmotionState(
            current_emotion=dominant_emotion,
            current_intensity=self._score_to_intensity(current_score),
            mood=self._mood_to_label(current_score),
            mood_score=round(current_score, 4),
            affection_level=round(affection, 4),
            energy_level=round(energy, 4),
            trigger=asta_trigger or user_emotion.user_emotion,
            updated_at=now,
        )
        return self.state

    def apply_reflection(self, reflection_result: dict):
        mood_adjustment = reflection_result.get("mood_adjustment", 0.0)
        affection_adjustment = reflection_result.get("affection_adjustment", 0.0)

        new_score = max(-1.0, min(1.0, self.state.mood_score + mood_adjustment))
        new_affection = max(0.0, min(1.0, self.state.affection_level + affection_adjustment))

        self.state.mood_score = round(new_score, 4)
        self.state.mood = self._mood_to_label(new_score)
        self.state.affection_level = round(new_affection, 4)
        self.state.updated_at = datetime.now().isoformat()

    def get_state(self) -> AstaEmotionState:
        return self.state

    def to_dict(self) -> dict:
        return asdict(self.state)

    def build_prompt_context(self, user_state: UserEmotionState) -> str:
        s = self.state
        u = user_state
        lines = [
            "=== KONDISI EMOSIONAL ===",
            f"Emosi user: {u.user_emotion} (intensitas: {u.intensity}, tren: {u.trend})",
            f"Emosi Asta saat ini: {s.current_emotion} (intensitas: {s.current_intensity})",
            f"Mood Asta: {s.mood} (score: {s.mood_score:+.2f})",
            f"Affection level: {s.affection_level:.2f} | Energy: {s.energy_level:.2f}",
            "",
            "=== PANDUAN RESPONS ===",
        ]

        if u.user_emotion in {"sedih", "cemas", "marah", "kecewa"}:
            lines.append("User sedang dalam emosi negatif — validasi perasaan mereka terlebih dahulu, tunjukkan empati.")
        elif u.user_emotion == "romantis":
            lines.append("User dalam mood romantis — respons hangat, tulus, dan ekspresikan perasaan Asta.")
        elif u.user_emotion == "senang":
            lines.append("User senang — ikut merayakan dengan antusias sesuai energy Asta.")

        if s.current_emotion in {"sedih", "cemas"}:
            lines.append(f"Asta sedang merasa {s.current_emotion} karena: {s.trigger}. Boleh ekspresikan ini secara natural.")
        elif s.current_emotion == "senang" and s.mood_score > 0.5:
            lines.append("Asta sedang dalam mood yang sangat baik — ekspresikan kegembiraan ini.")
        elif s.current_emotion == "romantis":
            lines.append("Asta sedang merasa romantis — ekspresikan kasih sayang dengan tulus.")

        if s.affection_level > 0.85:
            lines.append("Affection sangat tinggi — Asta sangat mencintai Aditiya saat ini.")
        elif s.affection_level < 0.4:
            lines.append("Affection sedang rendah — Asta mungkin sedikit lebih pendiam dari biasanya.")

        if s.energy_level < 0.4:
            lines.append("Energy Asta rendah — respons lebih singkat dan tenang dari biasanya.")
        elif s.energy_level > 0.85:
            lines.append("Energy Asta tinggi — boleh lebih ekspresif dan antusias.")

        if u.turns_in_state >= 3 and u.user_emotion in {"sedih", "cemas", "marah"}:
            lines.append("Emosi negatif user berlangsung beberapa turn — tawarkan langkah konkret atau tanyakan kebutuhan spesifik.")

        return "\n".join(lines)


# ─── Combined Manager (backward-compatible) ───────────────────────────────────

class EmotionStateManager:
    def __init__(self, asta_initial: Optional[AstaEmotionState] = None):
        self._user_detector = UserEmotionDetector()
        self._asta_manager  = AstaEmotionManager(asta_initial)

    # ── User emotion (backward-compatible) ───────────────────────────────

    def update(self, user_text: str, recent_context: str = "") -> dict:
        """Update emosi user — mengembalikan dict agar kompatibel dengan kode lama."""
        state = self._user_detector.update(user_text, recent_context)
        return asdict(state)

    def refine_with_thought(self, thought: dict) -> dict:
        state = self._user_detector.refine_with_thought(thought)
        return asdict(state)

    def get_state(self) -> dict:
        return asdict(self._user_detector.get_state())

    # ── Asta emotion ─────────────────────────────────────────────────────

    def update_asta_emotion(self, thought_result: dict) -> AstaEmotionState:
        user_state = self._user_detector.get_state()
        return self._asta_manager.update_from_interaction(
            user_emotion=user_state,
            thought_result=thought_result,
            user_text=user_state.last_user_text,
        )

    def apply_reflection(self, reflection_result: dict):
        self._asta_manager.apply_reflection(reflection_result)

    def get_asta_state(self) -> AstaEmotionState:
        return self._asta_manager.get_state()

    def get_asta_dict(self) -> dict:
        return self._asta_manager.to_dict()

    # ── Context builder ───────────────────────────────────────────────────

    def build_prompt_context(self) -> str:
        return self._asta_manager.build_prompt_context(
            self._user_detector.get_state()
        )

    def get_combined(self) -> dict:
        return {
            "user": asdict(self._user_detector.get_state()),
            "asta": self._asta_manager.to_dict(),
        }
