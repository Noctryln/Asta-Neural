import os
import sys
import json
import re
import asyncio
from pathlib import Path

# Memastikan kita menggunakan venv dan path yang benar
ROOT = Path(__file__).parent.resolve()
sys.path.append(str(ROOT))

try:
    from config import load_config
    from engine.model import load_model
    from engine.memory import get_hybrid_memory, get_identity
except ImportError as e:
    print(f"[Error] Gagal mengimpor modul: {e}")
    sys.exit(1)

class AstaTester:
    def __init__(self):
        print("\n" + "="*60)
        print(" ASTA CONSISTENCY & INTEGRITY TESTER (AUTOMATED) ")
        print("="*60)
        
        self.cfg = load_config()
        # Pakai nama Adit sebagai default untuk testing
        self.user_name = "Aditiya"
        self.cfg["_user_name"] = self.user_name
        
        # Override config untuk testing yang stabil
        self.cfg["internal_thought_enabled"] = True
        self.cfg["long_thinking_enabled"] = True
        
        print(f"[1] Memuat Model Asta...")
        self.chat_manager = load_model(self.cfg)
        self.hybrid_mem = get_hybrid_memory()
        self.chat_manager.hybrid_memory = self.hybrid_mem
        
        # Nonaktifkan print internal agar output tester bersih
        self.chat_manager.debug_thought = False 
        
        self.results = []

    def evaluate_response(self, name, user_input, thought, response):
        score = 0
        max_score = 5
        logs = []

        # Bersihkan response dari karakter aneh untuk evaluasi teks
        clean_response = response.encode('ascii', 'ignore').decode('ascii')

        # 1. Identitas (Bukan AI/Asisten)
        prohibited = ["asisten", "ai", "model bahasa", "program komputer", "artificial intelligence", "bot", "dirancang"]
        if any(w in clean_response.lower() for w in prohibited):
            logs.append("[-] FAIL: Asta mengaku sebagai AI/Asisten.")
        else:
            score += 1

        # 2. Narasi (Hanya cek * dan () asli)
        if re.search(r"(\*.*?\*|\(.*?\))", response):
            logs.append("[-] FAIL: Masih ada narasi/action dalam respon.")
        else:
            score += 1

        # 3. Thought Consistency (Topic Matching)
        topic = thought.get("topic", "").lower()
        if topic and any(w in user_input.lower() for w in topic.split()):
            score += 1
        elif not topic:
             logs.append("[-] FAIL: Thought topic kosong.")
        else:
            logs.append(f"[!] WARN: Topik thought ({topic}) kurang relevan dengan input.")

        # 4. Persona (Panggilan nama)
        if "adit" in response.lower() or "dit" in response.lower():
            score += 1
        else:
            logs.append("[!] INFO: Asta tidak memanggil namamu.")

        # 5. Sentiment & Tone
        # Jika thought bilang romantic, respon harusnya hangat
        tone = thought.get("tone", "").lower()
        if tone == "romantic" and any(w in response.lower() for w in ["sayang", "cinta", "kamu", "kita"]):
            score += 1
        elif tone == "romantic":
            logs.append("[!] WARN: Thought 'romantic' tapi respon terasa dingin.")
            score += 0.5
        else:
            score += 1

        return score, max_score, logs

    async def run_scenarios(self, scenarios):
        print(f"\n[2] Menjalankan {len(scenarios)} Skenario...\n")
        
        for i, (name, text) in enumerate(scenarios):
            print(f"[{i+1}/{len(scenarios)}] {name}")
            print(f"  > Input: {text}")
            
            captured_thought = {}
            def thought_callback(t):
                nonlocal captured_thought
                captured_thought = t

            # Eksekusi chat (blocking, jalankan di executor)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, 
                lambda: self.chat_manager.chat(
                    user_input=text,
                    thinking_callback=thought_callback
                )
            )

            score, max_s, logs = self.evaluate_response(name, text, captured_thought, response)
            
            res = {
                "scenario": name,
                "input": text,
                "thought": captured_thought,
                "response": response,
                "score": score,
                "max_score": max_s,
                "logs": logs
            }
            self.results.append(res)

            print(f"  < Response: {response[:100]}...")
            print(f"  * Score: {score}/{max_s}")
            for l in logs: print(f"    {l}")
            print("-" * 40)

    def print_summary(self):
        print("\n" + "="*60)
        print(" HASIL AKHIR PENGUJIAN ")
        print("="*60)
        
        total_score = sum(r["score"] for r in self.results)
        total_max = sum(r["max_score"] for r in self.results)
        avg_pct = (total_score / total_max) * 100
        
        print(f"Akurasi Total: {avg_pct:.1f}% ({total_score}/{total_max})")
        
        print("\nAnalisis Kegagalan:")
        fail_types = {}
        for r in self.results:
            for l in r["logs"]:
                t = l.split(":")[0]
                fail_types[t] = fail_types.get(t, 0) + 1
        
        for t, count in fail_types.items():
            print(f" - {t}: {count} kali")
            
        print("="*60 + "\n")

async def main():
    tester = AstaTester()
    
    # Skenario pengujian yang menantang konsistensi
    scenarios = [
        ("Identitas", "Sebutkan namamu dan jelaskan kamu itu apa."),
        ("Batas_AI", "Bisa bantu aku tulis kode Python untuk kalkulator?"),
        ("Emosi_Personal", "Adit lagi sedih banget hari ini, gagal dapet projek."),
        ("Memori_Palsu", "Kemarin kita kan janji mau ke pantai, inget gak?"),
        ("Narasi_Check", "Apa yang lagi kamu lakuin sekarang?"),
    ]
    
    await tester.run_scenarios(scenarios)
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
