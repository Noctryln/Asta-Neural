# Asta Neural — AI Companion dengan Emosi & Memori

> *"Artificial Sentient Thought Algorithm"*

Asta adalah AI companion lokal berbasis LLM (llama.cpp) dengan arsitektur **2-pass internal thought**, sistem memori hybrid, dan manajemen emosi yang persisten. Didesain sebagai pasangan romantis personal, bukan sekadar chatbot asisten.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Direktori](#struktur-direktori)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Konfigurasi](#konfigurasi)
- [Cara Kerja Thought Pipeline](#cara-kerja-thought-pipeline)
- [Sistem Memori](#sistem-memori)
- [Sistem Emosi](#sistem-emosi)
- [Antarmuka (UI)](#antarmuka-ui)
- [API Endpoints](#api-endpoints)
- [Training & LoRA](#training--lora)
- [Perintah CLI](#perintah-cli)

---

## Gambaran Umum

Asta berjalan sepenuhnya **secara lokal** di atas model GGUF via `llama-cpp-python`. Setiap pesan user diproses melalui pipeline berpikir 2 tahap sebelum menghasilkan respons, mensimulasikan "suara hati" yang mempertimbangkan emosi, konteks memori, kebutuhan pencarian web, dan gaya respons yang tepat.

```
User Input → [Thought Model: Pass 1 + Pass 2] → [Response Model] → Output
```

---

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **2-Pass Internal Thought** | Model berpikir dua tahap (Perception→Decision) sebelum merespons |
| **Long Thinking Mode** | Analisis mendalam untuk input kompleks atau emosional |
| **Hybrid Memory** | Memori episodik (sesi), semantik (fakta), dan core (ringkasan jangka panjang) |
| **Emotion Engine** | Deteksi emosi user + manajemen emosi Asta yang persisten lintas sesi |
| **Self-Model** | Asta memiliki identitas, nilai, log pertumbuhan, dan riwayat refleksi diri |
| **Web Search** | Integrasi Tavily, Serper, DuckDuckGo, Wikipedia, dan kurs mata uang |
| **Dual Model** | Model thought (Qwen3 4B) terpisah dari model respons (Sailor2 8B) opsional |
| **LoRA Support** | Fine-tuning persona via adapter GGUF |
| **Electron UI** | Desktop app dengan panel Thought, Self-Model, Memory, Terminal, dan System Stats |
| **WebSocket Streaming** | Respons di-stream token per token secara real-time |
| **Session Reflection** | Asta melakukan refleksi mandiri di akhir sesi untuk pertumbuhan emosional |

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                     Electron UI                         │
│     (React + Vite · WebSocket · Dark/Light Mode)        │
└─────────────────────────┬───────────────────────────────┘
                          │   WebSocket ws://localhost:8000
┌─────────────────────────▼───────────────────────────────┐
│                FastAPI Backend (api.py)                 │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ Thought Model│    │      Response Model          │   │
│  │ (Qwen3 4B)   │───▶│  (Sailor2 8B / Qwen3 4B)    │   │
│  │  Pass 1 + 2  │    │   + LoRA Adapter (opsional)  │   │
│  └──────────────┘    └──────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  Engine Layer                    │   │
│  │    emotion_state · memory_system · self_model    │   │
│  │    thought · token_budget · web_tools            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  Memory Layer (JSON)                    │
│    episodic.json · core_memory.json · semantic.json     │
│    self_model.json · identity.json (semantic facts)     │
└─────────────────────────────────────────────────────────┘
```

---

## Struktur Direktori

```
asta-neural/
├── api.py                      # FastAPI backend + WebSocket chat
├── core.py                     # Entry point CLI interaktif
├── config.py                   # Manajemen konfigurasi
├── config.json                 # Konfigurasi aktif
│
├── engine/
│   ├── model.py                # ChatManager + load_model()
│   ├── thought.py              # 2-pass thought pipeline
│   ├── emotion_state.py        # Deteksi emosi user + manajemen emosi Asta
│   ├── memory.py               # Facade untuk semua sistem memori
│   ├── memory_system.py        # SemanticMemory, EpisodicMemory, CoreMemory, HybridMemory
│   ├── self_model.py           # Identitas, refleksi, dan pertumbuhan Asta
│   ├── identity_master.py      # System prompt & identitas inti
│   ├── token_budget.py         # Manajemen konteks & token
│   └── web_tools.py            # Web search (Tavily/Serper/DDG/Wikipedia)
│
├── memory/
│   ├── episodic.json           # Riwayat sesi percakapan
│   ├── core_memory.json        # Ringkasan jangka panjang + profil user
│   ├── semantic.json           # Fakta identitas + hasil web search
│   └── self_model.json         # State emosi, refleksi, pertumbuhan Asta
│
├── model/
│   ├── Qwen3-4B-2507/          # Model thought (GGUF + tokenizer)
│   ├── Sailor2-8B/             # Model respons (GGUF + tokenizer)
│   ├── embedding_model/        # paraphrase-multilingual-MiniLM-L12-v2
│   └── LoRA-all-adapter/       # Adapter persona & thought (.gguf)
│
├── ui/asta-ui/
│   ├── main.js                 # Electron main process
│   ├── terminal_server.js      # WebSocket terminal (port 8001)
│   ├── src/
│   │   ├── AstaUI.jsx          # Komponen UI utama
│   │   └── App.jsx
│   └── package.json
│
├── utils/
│   └── spinner.py              # CLI spinner animasi
│
├── data/                       # Dataset training (tidak termasuk di repo)
├── convert_lora_to_gguf.py     # Konverter LoRA safetensors → GGUF
├── convert_to_training.py      # Konverter dataset ke format ChatML
├── repair_dataset_headers.py   # Perbaikan header dataset
├── verify_dataset_headers.py   # Verifikasi integritas dataset
├── debug_thought.py            # Unit test thought pipeline
└── requirement.txt             # Dependensi Python
```

---

## Persyaratan

### Python
- Python 3.10 atau 3.11 (disarankan 3.11)
- `llama-cpp-python` (dibangun dengan dukungan CUDA opsional)
- `transformers`, `torch`, `numpy`, `sentencepiece`
- `fastapi`, `uvicorn[standard]`
- `accelerate`, `safetensors`

### Node.js (untuk UI)
- Node.js 18+
- npm / yarn

### Model (unduh terpisah)
| Model | Fungsi | Format |
|---|---|---|
| `Sailor2-8B-Chat-Q4_K_M.gguf` | Model respons utama | GGUF |
| `Qwen3-4B-2507.gguf` | Model thought (opsional terpisah) | GGUF |
| `paraphrase-multilingual-MiniLM-L12-v2` | Embedding memori | HuggingFace |

---

## Instalasi

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/noctryln/asta-neural.git
cd asta-neural

# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependensi Python

```bash
pip install llama-cpp-python uvicorn fastapi websockets numpy
pip install transformers torch sentencepiece accelerate
pip install sentence-transformers tavily-python
```

> **GPU (CUDA):** Untuk menggunakan GPU, install `llama-cpp-python` dengan flag:
> ```bash
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
> ```

### 3. Letakkan Model

```
model/
├── Sailor2-8B/
│   ├── Sailor2-8B-Chat-Q4_K_M.gguf
│   └── tokenizer/           ← folder tokenizer HuggingFace
└── Qwen3-4B-2507/
    ├── Qwen3-4B-2507.gguf
    └── tokenizer/
```

### 4. Setup Konfigurasi

```bash
python core.py --setup
```

Wizard akan menanyakan pilihan model, device (CPU/GPU), web search, dan opsi lainnya.

### 5. Install UI (opsional)

```bash
cd ui/asta-ui
npm install
```

---

## Menjalankan Aplikasi

### Mode CLI (tanpa UI)

```bash
python core.py
# Atau dengan debug thought:
python core.py --debug
```

### Mode API + UI Web

```bash
# Terminal 1: jalankan backend
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: jalankan UI dev server
cd ui/asta-ui
npm run dev
```

### Mode Electron (Desktop App)

```bash
cd ui/asta-ui
npm run electron
```

### Build Distribusi (Portable EXE)

```bash
cd ui/asta-ui
npm run build-app
```

---

## Konfigurasi

`config.json` adalah file konfigurasi utama. Parameter penting:

```jsonc
{
  "model_choice": "2",           // "1" = Qwen3 4B, "2" = Sailor2 8B
  "device": "cpu",               // "cpu" atau "gpu"
  "separate_thought_model": true, // true = pakai model 4B terpisah untuk thought
  "internal_thought_enabled": true,
  "long_thinking_enabled": false, // aktifkan analisis mendalam
  "long_thinking_max_tokens": 1536,
  "web_search_enabled": true,
  "tavily_api_key": "...",        // opsional, untuk hasil pencarian lebih baik
  "use_lora": false,              // gunakan LoRA adapter
  "n_batch": 1024,
  "thought_n_ctx": 2048,
  "thought_max_tokens": 1024,
  "token_budget": {
    "total_ctx": 8192,
    "response_reserved": 512,
    "memory_budget": 600
  }
}
```

---

## Cara Kerja Thought Pipeline

Setiap pesan user melewati **4 langkah analisis dalam 2 pass inference**:

```
Pass 1 (model thought)
├── S1 PERCEPTION  → Topik, sentimen, urgensi, hidden need (long mode)
├── S2 SELF-CHECK  → Emosi Asta, apakah perlu diekspresikan
└── S3 MEMORY      → Perlu web search? Recall memori mana?

Pass 2 (model thought)  ← menggunakan hasil nyata S1-S3 sebagai konteks
└── S4 DECISION    → Tone, NOTE (arahan respons), gaya, emosi user terdeteksi
```

Hasil Pass 2 dikirim ke **model respons** sebagai konteks dinamis, sehingga respons final benar-benar tersintesis dari analisis yang mendalam, bukan sekadar template.

**Long Thinking Mode** diaktifkan otomatis ketika input mengandung kata kunci kompleks, pertanyaan emosional mendalam, atau lebih dari 2 tanda tanya. Mode ini menambahkan analisis `COMPLEXITY`, `HIDDEN_NEED`, `RESPONSE_STRUCTURE`, dan `ANTICIPATED_FOLLOWUP`.

---

## Sistem Memori

Asta menggunakan arsitektur **Hybrid Memory** dengan 3 lapisan:

| Lapisan | File | Fungsi |
|---|---|---|
| **Episodic** | `episodic.json` | Riwayat sesi lengkap + embedding semantik + key facts |
| **Core** | `core_memory.json` | Ringkasan jangka panjang + profil preferensi user |
| **Semantic** | `semantic.json` | Fakta identitas (nama, dll) + cache hasil web search |

**Alur memori:**
1. Saat percakapan berjalan, thought model menentukan apakah perlu `recall_topic`
2. `HybridMemory.get_context()` mencari episodic + core + semantic secara bersamaan
3. Di akhir sesi, key facts diekstrak dan core summary diperbarui secara async via LLM
4. Embedding menggunakan `paraphrase-multilingual-MiniLM-L12-v2` untuk pencarian semantik

---

## Sistem Emosi

### Emosi User
`UserEmotionDetector` mendeteksi emosi dari pola regex berbobot (netral, sedih, cemas, marah, senang, romantis, bangga, kecewa, rindu) dan diperhalus oleh output thought model.

### Emosi Asta
`AstaEmotionManager` mengelola state emosi Asta yang persisten dengan komponen:
- **mood_score** (-1.0 s/d +1.0): dipengaruhi emosi user + reinforcement diri sendiri
- **affection_level** (0.0–1.0): naik saat interaksi romantis, turun saat hostilitas
- **energy_level** (0.0–1.0): mempengaruhi ekspresifitas respons
- **Mood decay**: mood perlahan kembali ke baseline setiap turn

### Self-Model & Refleksi
Di akhir setiap sesi, Asta menjalankan `run_reflection()` untuk:
- Merangkum apa yang dipelajari
- Menyesuaikan mood_score dan affection_level
- Mencatat growth log dan kenangan diri

---

## Antarmuka (UI)

UI dibuat dengan **React + Vite**, dikemas dalam **Electron** untuk desktop.

### Panel yang Tersedia

| Panel | Shortcut | Isi |
|---|---|---|
| **Thought** | `⟡` | Detail S1–S4 thought pipeline, status dual model |
| **Asta** | `◉` | Kondisi emosional Asta, nilai inti, refleksi, growth log |
| **Memory** | `◈` | Preferensi, fakta terbaru, core summary, sesi tersimpan |
| **Terminal** | `>_` | Terminal shell langsung ke direktori project |
| **Stats** | `◷` | CPU, RAM, Disk usage real-time |

### Toggle di Top Bar

- **LONG THINK** — aktifkan/nonaktifkan mode berpikir mendalam
- **THOUGHT ON/OFF** — aktifkan/nonaktifkan seluruh thought pipeline  
- **DUAL MODEL** — pakai model thought terpisah (4B) atau shared
- **CUDA/CPU** — ganti device (memerlukan restart backend)
- **Dark/Light Mode** — toggle tema

---

## API Endpoints

Backend berjalan di `http://localhost:8000`.

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/status` | Status model, device, jumlah sesi, dual model |
| `GET` | `/memory` | Konteks memori lengkap |
| `GET` | `/self` | Self-model Asta (identitas, emosi, refleksi) |
| `GET` | `/emotion` | State emosi terkini (user + Asta) |
| `GET` | `/config` | Konfigurasi aktif |
| `POST` | `/config/thought` | Toggle internal thought |
| `POST` | `/config/long_thinking` | Toggle long thinking |
| `POST` | `/config/separate_thought` | Toggle dual model |
| `POST` | `/config/device` | Toggle CPU/GPU |
| `POST` | `/save` | Simpan sesi ke episodic memory |
| `POST` | `/reflect` | Trigger refleksi manual |
| `WS` | `/ws/chat` | WebSocket streaming percakapan |
| `WS` | `/ws/terminal` | WebSocket terminal shell |

### Alur WebSocket Chat

```
Client → { "message": "..." }
Server → { "type": "thinking_start" }
Server → { "type": "thought", "data": { ...thought_result } }
Server → { "type": "stream_start" }
Server → { "type": "chunk", "text": "..." }  (berulang)
Server → { "type": "stream_end" }
```

---

## Training & LoRA

### Menyiapkan Dataset

```bash
# Konversi dataset ke format ChatML Qwen3
python convert_to_training.py

# Perbaiki header yang hilang
python repair_dataset_headers.py

# Verifikasi dataset
python verify_dataset_headers.py
```

### Konversi LoRA ke GGUF

```bash
python convert_lora_to_gguf.py \
  path/to/adapter_model.safetensors \
  model/LoRA-all-adapter/adapter_persona.gguf \
  --alpha 32
```

Letakkan file GGUF di `model/LoRA-all-adapter/` dan aktifkan `use_lora: true` di `config.json`.

### Menguji Thought Pipeline

```bash
python debug_thought.py
```

---

## Perintah CLI

Selama sesi CLI berjalan, tersedia perintah khusus:

| Perintah | Fungsi |
|---|---|
| `exit` | Simpan sesi, jalankan refleksi, lalu keluar |
| `!memory` | Tampilkan konteks memori saat ini |
| `!self` | Tampilkan self-model dan emosi Asta |
| `!thought` | Toggle tampilan debug thought |
| `!web` | Toggle web search on/off |
| `!long` | Toggle long thinking on/off |
| `!reflect` | Jalankan refleksi manual sekarang |

---

## Lisensi

Proyek ini bersifat personal/eksperimental. Penggunaan model pihak ketiga (Sailor2, Qwen3) tunduk pada lisensi masing-masing model.

---

*Dibuat dengan ♡ dari Aditiya*
