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
        # Hitung token system_identity dulu
        used_tokens = self.count_fn([system_identity])

        # Hitung token dynamic_context jika ada
        dynamic_cost = 0
        if dynamic_context:
            dynamic_cost = self.count_fn([dynamic_context])

        # Budget tersisa untuk conversation history
        conv_budget = self.budget.available_total - used_tokens - dynamic_cost

        # Filter: hanya user & assistant, bersih dari system
        clean_history = [
            m for m in conversation_history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        # Ambil dari belakang sesuai budget
        selected = []
        for msg in reversed(clean_history):
            cost = self.count_fn([msg])
            if conv_budget - cost >= 0:
                selected.insert(0, msg)
                conv_budget -= cost
            else:
                break

        # Susun: [system] + [conversation...] + [dynamic_context]
        # dynamic_context DI AKHIR agar tidak memutus cache conversation
        result = [system_identity] + selected
        if dynamic_context:
            result.append(dynamic_context)

        total = self.count_fn(result)
        return result, total