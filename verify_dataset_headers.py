import json
import os

INPUT_FILE = "data/asta_thought_training_qwen_v2.json"

def verify_headers():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} tidak ditemukan!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Gagal membaca JSON: {e}")
            return

    print(f"Memverifikasi {len(data)} sampel training...\n")

    missing_header_count = 0
    fail_samples = []

    for i, item in enumerate(data):
        text = item.get("text", "")
        
        if "<|im_start|>assistant" not in text:
            missing_header_count += 1
            fail_samples.append({"index": i, "issues": ["No Assistant Tag"], "preview": text[-50:]})
            continue
            
        parts = text.split("<|im_start|>assistant")
        assistant_part = parts[-1].split("<|im_end|>")[0].strip()
        
        content = assistant_part
        
        is_pass1 = "TOPIC:" in content
        is_pass2 = "TONE:" in content
        
        issues = []
        
        if is_pass1:
            if not ("STEP 1" in content or "FASE 1" in content): issues.append("Missing S1 Header")
            if not ("STEP 2" in content or "FASE 2" in content): issues.append("Missing S2 Header")
            if not ("STEP 3" in content or "FASE 3" in content): issues.append("Missing S3 Header")
        elif is_pass2:
            if not ("STEP 4" in content or "FASE 4" in content): issues.append("Missing S4 Header")
        else:
            issues.append("Unknown Structure (No TOPIC/TONE)")

        if issues:
            missing_header_count += 1
            fail_samples.append({
                "index": i,
                "issues": issues,
                "preview": content[:100].replace("\n", " ") + "..."
            })

    # Laporan
    if missing_header_count == 0:
        print("Semua sampel memiliki header yang lengkap.")
    else:
        print(f"Ditemukan {missing_header_count} sampel bermasalah dari {len(data)} total.")
        print("\nContoh sampel gagal (10 pertama):")
        for fail in fail_samples[:10]:
            print(f"- Index {fail['index']}: {', '.join(fail['issues'])}")
            print(f"  Preview: {fail['preview']}\n")

if __name__ == "__main__":
    verify_headers()
