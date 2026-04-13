import json
import os
import re

INPUT_FILE = "data/asta_thought_training_qwen_v2.json"
OUTPUT_FILE = "data/asta_thought_training_qwen_v2.json"
def repair_headers():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} tidak ditemukan!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Memulai perbaikan pada {len(data)} sampel...")
    repaired_count = 0

    for i, item in enumerate(data):
        text = item.get("text", "")
        if "<|im_start|>assistant" not in text: continue

        # Pisahkan komponen ChatML
        parts = text.split("<|im_start|>assistant")
        header_user_part = parts[0]
        assistant_content = parts[1].split("<|im_end|>")[0].strip()
        footer_part = parts[1].split("<|im_end|>")[1] if "<|im_end|>" in parts[1] else ""

        # Deteksi Jenis (Long vs Normal)
        is_long = "FASE" in header_user_part or "ANALISIS MENDALAM" in header_user_part
        is_pass1 = "TOPIC:" in assistant_content
        is_pass2 = "TONE:" in assistant_content

        original_content = assistant_content
        
        if is_pass1:
            # Repair Pass 1 (S1-S3)
            if "STEP 1" not in assistant_content and "FASE 1" not in assistant_content:
                h = "=== FASE 1: DEEP PERCEPTION ===" if is_long else "=== STEP 1: PERCEPTION ==="
                assistant_content = h + "\n" + assistant_content
            
            if "STEP 2" not in assistant_content and "FASE 2" not in assistant_content:
                h = "=== FASE 2: DEEP SELF-CHECK ===" if is_long else "=== STEP 2: SELF-CHECK ==="
                assistant_content = assistant_content.replace("ASTA_EMOTION:", f"\n\n{h}\nASTA_EMOTION:")
            
            if "STEP 3" not in assistant_content and "FASE 3" not in assistant_content:
                h = "=== FASE 3: CONTEXT ANALYSIS ===" if is_long else "=== STEP 3: MEMORY & SEARCH ==="
                assistant_content = assistant_content.replace("REASONING:", f"\n\n{h}\nREASONING:")

        elif is_pass2:
            # Repair Pass 2 (S4)
            if "STEP 4" not in assistant_content and "FASE 4" not in assistant_content:
                h = "=== FASE 4: RESPONSE PLANNING ===" if is_long else "=== STEP 4: DECISION ==="
                assistant_content = h + "\n" + assistant_content

        if assistant_content != original_content:
            new_text = f"{header_user_part}<|im_start|>assistant\n{assistant_content}\n<|im_end|>{footer_part}"
            item["text"] = new_text
            repaired_count += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Berhasil memperbaiki {repaired_count} sampel.")

if __name__ == "__main__":
    repair_headers()
