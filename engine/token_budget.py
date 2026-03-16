from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TokenBudget:
    total_ctx: int = 8192
    response_reserved: int = 512
    system_identity: int = 350
    memory_budget: int = 600

    @property
    def conversation_budget(self) -> int:
        return (
            self.total_ctx
            - self.response_reserved
            - self.system_identity
            - self.memory_budget
        )

    @property
    def available_total(self) -> int:
        return self.total_ctx - self.response_reserved


class TokenBudgetManager:
    def __init__(self, budget: TokenBudget, count_fn):
        self.budget   = budget
        self.count_fn = count_fn

    def build_messages(
        self,
        system_identity: Dict,
        memory_messages: List[Dict],      # tidak dipakai, dipertahankan compat
        conversation_history: List[Dict],
        dynamic_context: Optional[Dict] = None,
    ) -> tuple:
        """
        Susun messages dengan urutan yang BENAR untuk memaksimalkan KV cache.

        llama.cpp melakukan prefix-match secara LINEAR dari token pertama.
        Begitu ada satu token berbeda, semua token setelahnya harus re-eval.

        URUTAN OPTIMAL:
          [0]   system_identity    ← konstan setiap turn → selalu cache hit
          [1]   user_turn_1        ← konstan sejak turn 1 → cache hit ab turn 2
          [2]   assistant_turn_1   ← konstan → cache hit
          ...
          [N-1] user_turn_N        ← turn ini (baru)
          [N]   dynamic_context    ← TERAKHIR, tepat sebelum asisten menjawab
                                     Berubah tiap turn tapi di akhir — tidak
                                     memutus cache conversation di atasnya

        Dengan urutan ini:
          Turn 1: 0 cache hit (semua baru)
          Turn 2: hit untuk [0]+[1]+[2] = system + turn1_user + turn1_assistant
          Turn 3: hit untuk [0]+[1]+[2]+[3]+[4] = system + turn1 + turn2
          ...dan seterusnya bertambah ~2 pesan per turn

        CATATAN: conversation_history HARUS hanya berisi role:user dan role:assistant.
        Jangan pernah menyimpan role:system di conversation_history.
        """
        # Filter: hanya user & assistant, bersih dari system
        clean_history = [
            m for m in conversation_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        # Strategi stabil untuk KV cache + relevansi:
        # 1) Selalu pertahankan ekor percakapan (tail) agar konteks terbaru masuk.
        # 2) Isi sisa budget dari awal percakapan (head) agar prefix tetap stabil.
        # Dengan ini, perubahan dynamic_context tidak mudah menggeser awal prompt,
        # sehingga prefix-match tidak mudah drop drastis.
        max_prompt_tokens = self.budget.available_total
        tail_keep = 4

        def _build_prompt(msgs: List[Dict]) -> List[Dict]:
            result = [system_identity] + msgs
            if dynamic_context:
                result.append(dynamic_context)
            return result

        # Pastikan message terbaru tetap ada.
        tail = clean_history[-tail_keep:] if len(clean_history) > tail_keep else clean_history[:]
        if self.count_fn(_build_prompt(tail)) > max_prompt_tokens:
            # Fallback keras: tail terlalu besar, ambil secukupnya dari belakang.
            selected = []
            for msg in reversed(clean_history):
                trial_selected = [msg] + selected
                if self.count_fn(_build_prompt(trial_selected)) <= max_prompt_tokens:
                    selected = trial_selected
                else:
                    break
        else:
            head_candidates = clean_history[:-len(tail)] if tail else clean_history
            selected = tail[:]

            # Isi dari depan agar prefix antar-turn stabil.
            for msg in head_candidates:
                trial_selected = selected[:-len(tail)] + [msg] + tail if tail else selected + [msg]
                if self.count_fn(_build_prompt(trial_selected)) <= max_prompt_tokens:
                    selected = trial_selected
                else:
                    break

        # Susun: [system] + [conversation...] + [dynamic_context]
        # dynamic_context DI AKHIR agar tidak memutus cache conversation
        result = _build_prompt(selected)

        total = self.count_fn(result)
        return result, total
