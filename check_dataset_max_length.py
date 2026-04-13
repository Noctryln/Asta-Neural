import json
import os
from transformers import AutoTokenizer

# Path konfigurasi
DATASET_PATH = "data/response_train_v4.json"
TOKENIZER_PATH = "model/Qwen3-8B/tokenizer"

def check_max_length():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset tidak ditemukan di {DATASET_PATH}")
        return

    print(f"Memuat tokenizer dari {TOKENIZER_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Gagal memuat tokenizer: {e}")
        print("Mencoba menggunakan tokenizer default (gpt2) sebagai estimasi...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Membaca dataset: {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    max_tokens = 0
    total_tokens = 0
    lengths = []

    print(f"Menganalisis {total_entries} entri...")

    for i, entry in enumerate(data):
        # Membaca field "text" secara langsung
        if "text" in entry:
            full_text = entry["text"]
        elif "messages" in entry:
            # Fallback jika ada format messages
            full_text = ""
            for msg in entry["messages"]:
                full_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        else:
            print(f"Warning: Entri ke-{i} tidak memiliki field 'text' atau 'messages'. Skip.")
            continue
            
        # Hitung token
        tokens = tokenizer.encode(full_text)
        length = len(tokens)
        
        lengths.append(length)
        if length > max_tokens:
            max_tokens = length
        total_tokens += length

        if (i + 1) % 500 == 0:
            print(f"  Sudah memproses {i + 1}/{total_entries} entri...")

    if not lengths:
        print("Tidak ada data yang valid untuk dianalisis.")
        return

    avg_tokens = total_tokens / total_entries
    lengths.sort()

    print("\n" + "="*40)
    print("HASIL ANALISIS DATASET (QWEN V2)")
    print("="*40)
    print(f"Total Entri      : {total_entries}")
    print(f"Max Length       : {max_tokens} token")
    print(f"Average Length   : {avg_tokens:.2f} token")
    print("-" * 40)
    print("PERSENTIL (Safe Padding Range):")
    print(f"90% data di bawah : {lengths[int(total_entries * 0.9)]} token")
    print(f"95% data di bawah : {lengths[int(total_entries * 0.95)]} token")
    print(f"99% data di bawah : {lengths[int(total_entries * 0.99)]} token")
    print("="*40)
    
    # Rekomendasi
    if max_tokens > 2048:
        print(f"Saran: Gunakan max_seq_length minimal {((max_tokens // 1024) + 1) * 1024}.")
    elif max_tokens > 1024:
        print("Saran: Gunakan max_seq_length minimal 2048.")
    else:
        print("Saran: Gunakan max_seq_length 1024 sudah cukup.")

if __name__ == "__main__":
    check_max_length()
