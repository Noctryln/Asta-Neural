import json
import os

INPUT_FILE = "data/response_v4.json"
OUTPUT_FILE = "data/response_train_v4.json"

def format_qwen_chatml(messages):
    """
    <|im_start|>system
    {content}<|im_end|>
    <|im_start|>user
    {content}<|im_end|>
    <|im_start|>assistant
    {content}<|im_end|>
    """
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        
        # Format ChatML standar Qwen
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return formatted_text.strip()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File {INPUT_FILE} tidak ditemukan!")
        return

    print(f"Konversi ke format ChatML...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError:
            print("File JSON corrupt atau kosong.")
            return

    training_data = []
    for entry in raw_data:
        if "messages" in entry:
            chatml_string = format_qwen_chatml(entry["messages"])
            training_data.append({"text": chatml_string})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Berhasil mengonversi {len(training_data)} sampel training.")
    print(f"File siap digunakan: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
