# Asta Neural — AI Companion dengan sistem emosi

> *"Artificial Sentient Thought Algorithm"*
> AI companion lokal berbasis LLM dengan arsitektur 2-pass internal thought, sistem emosi persisten, dan memori hybrid multi-lapisan.

---

## Daftar Isi

1. [Gambaran Arsitektur Sistem](#gambaran-arsitektur-sistem)
2. [Thought Pipeline — Inti Kecerdasan Asta](#thought-pipeline--inti-kecerdasan-asta)
3. [Sistem Emosi — Dual-Layer Emotion Engine](#sistem-emosi--dual-layer-emotion-engine)
4. [Hybrid Memory Architecture](#hybrid-memory-architecture)
5. [Self-Model & Refleksi](#self-model--refleksi)
6. [Token Budget Management](#token-budget-management)
7. [Model Loading & Dual-Model Architecture](#model-loading--dual-model-architecture)
8. [FastAPI Backend & WebSocket Protocol](#fastapi-backend--websocket-protocol)
9. [Frontend — Electron + React UI](#frontend--electron--react-ui)
10. [Dataset Generation Pipeline](#dataset-generation-pipeline)
11. [LoRA Fine-tuning Pipeline](#lora-fine-tuning-pipeline)
12. [Web Search Integration](#web-search-integration)
13. [Persyaratan & Instalasi](#persyaratan--instalasi)
14. [Konfigurasi Lengkap](#konfigurasi-lengkap)
15. [API Reference](#api-reference)
16. [Alur Data End-to-End](#alur-data-end-to-end)

---

## Gambaran Arsitektur Sistem

Asta Neural adalah sistem AI companion yang dirancang di atas prinsip bahwa setiap respons yang dihasilkan harus melewati proses introspeksi eksplisit sebelum disampaikan ke pengguna. Berbeda dari chatbot konvensional yang langsung memasukkan input ke dalam model dan menghasilkan output, Asta menggunakan arsitektur **2-pass inference** di mana pass pertama menjalankan pipeline analitik internal (disebut "thought") dan pass kedua menggunakan hasil analisis tersebut sebagai konteks dinamis untuk menghasilkan respons yang koheren secara emosional dan kontekstual.

Secara keseluruhan, sistem ini terdiri dari lima lapisan besar yang saling berinteraksi:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Electron UI Layer                        │
│         React + Vite · WebSocket Streaming · Dark Mode          │
│   [Thought Panel] [Self Panel] [Memory Panel] [Terminal Panel]  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  WebSocket ws://localhost:8000
┌──────────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend  (api.py)                     │
│           WebSocket /ws/chat · REST /status /memory /self       │
│                      asyncio + _chat_lock                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    ChatManager  (engine/model.py)               │
│                                                                 │
│  ┌──────────────────────┐    ┌───────────────────────────────┐  │
│  │  Thought Model       │    │     Response Model            │  │
│  │  Qwen3-4B-2507.gguf  │───▶│  Qwen3-8B.gguf / Sailor2-8B  │  │
│  │  Pass 1 + Pass 2     │    │   + LoRA adapter              │  │
│  └──────────────────────┘    └───────────────────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Engine Subsystems                      │ │
│  │  EmotionStateManager · HybridMemory · SelfModel            │ │
│  │  TokenBudgetManager  · WebTools · ThoughtPipeline          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Persistence Layer (JSON)                    │
│   episodic.json · core_memory.json · semantic.json             │
│   self_model.json · config.json                                │
└─────────────────────────────────────────────────────────────────┘
```

Seluruh komponen berjalan secara **lokal** tanpa ketergantungan pada API cloud (kecuali opsional untuk web search). Model-model GGUF dijalankan via `llama-cpp-python`, dan embedding semantik menggunakan `paraphrase-multilingual-MiniLM-L12-v2` yang juga berjalan lokal.

---

## Thought Pipeline — Inti Kecerdasan Asta

Komponen paling kritis dalam sistem ini adalah thought pipeline yang didefinisikan di `engine/thought.py`. Pipeline ini adalah yang membedakan Asta dari chatbot biasa — setiap pesan user tidak langsung diolah oleh response model, melainkan terlebih dahulu dianalisis secara mendalam melalui empat tahap terstruktur yang dibagi dalam dua pass inference terpisah.

### Arsitektur 2-Pass

**Pass 1** menjalankan tiga tahap analisis pertama (S1–S3) secara berurutan dalam satu inferensi:

*STEP 1: PERCEPTION* — Model mengidentifikasi topik pembicaraan, sentimen keseluruhan (positif/negatif/netral), tingkat urgensi, dan — dalam Long Thinking Mode — kebutuhan tersembunyi (hidden need) yang tidak diungkapkan secara eksplisit oleh pengguna. Analisis ini menggunakan konteks dari riwayat percakapan terbaru dan memori lightweight.

*STEP 2: SELF-CHECK* — Model menganalisis kondisi emosional Asta sendiri sebagai respons terhadap input yang diterima. Ini mencakup identifikasi emosi dominan Asta (dari palet: netral, sedih, cemas, marah, senang, romantis, rindu, bangga, kecewa), pemicu emosi tersebut, dan keputusan apakah emosi tersebut perlu diekspresikan dalam respons.

*STEP 3: CONTEXT* — Model memutuskan kebutuhan konteks eksternal: apakah perlu melakukan web search (dan query apa yang digunakan), apakah perlu me-recall memori episodik tertentu (dan topik apa yang dicari), serta alasan di balik keputusan-keputusan tersebut (reasoning).

Hasil dari Pass 1 kemudian diringkas menjadi sebuah `s1_s2_s3_summary` yang berisi representasi kompak dari seluruh analisis ketiga tahap.

**Pass 2** menggunakan summary tersebut sebagai konteks tambahan untuk menjalankan tahap keempat:

*STEP 4: DECISION (REALISASI)* — Dengan mempertimbangkan hasil analisis S1–S3, model memutuskan tone respons (romantic/ceria/lembut/netral/tegas/malas), gaya respons, tingkat formalitas, deteksi emosi user yang telah diperhalus, confidence level deteksi emosi, dan yang paling penting: **NOTE** — sebuah inner-voice directive yang menjadi panduan utama bagi response model dalam menghasilkan jawaban akhir.

### Long Thinking Mode

Sistem secara otomatis mendeteksi apakah suatu input memerlukan analisis mendalam berdasarkan beberapa heuristik yang didefinisikan dalam fungsi `should_use_long_thinking()`:

- Kehadiran kata kunci kompleks seperti "kenapa", "mengapa", "jelaskan", "analisis", atau kata-kata yang mengindikasikan curhat mendalam
- Pola emosi berat seperti "sedih banget", "nangis", "hancur", atau "depresi"
- Panjang input melebihi 30 kata dengan kehadiran pertanyaan
- Lebih dari dua tanda tanya dalam satu pesan
- Permintaan eksplisit untuk analisis mendalam

Dalam Long Thinking Mode, template pass 1 diperluas dengan field tambahan: `SUBTOPIC`, `COMPLEXITY`, `HIDDEN_NEED`, `SOCIAL_HINT`, `CONVERSATIONAL_GOAL`, `CONTEXT_GAPS`, dan `MISSING_INFO`. Pass 2 juga diperluas dengan `RESPONSE_STRUCTURE` dan `ANTICIPATED_FOLLOWUP` yang membantu response model menyusun jawaban yang lebih terstruktur dan antisipatif.

### Safety Filters & Rule-Based Fallbacks

Pipeline ini juga mencakup beberapa mekanisme keamanan:

`_apply_safety_filter_search()` memblokir query web search yang berpotensi bersifat meta (mencari tentang "Asta", "AI", "model") atau yang terdeteksi sebagai keluhan tentang respons sebelumnya (bukan pencarian informasi genuine).

`_detect_repetition()` menganalisis overlap kata antara input saat ini dengan input sebelumnya untuk mendeteksi pola pengulangan yang mungkin mengindikasikan bahwa pengguna belum mendapatkan jawaban yang memuaskan.

`_check_escalation_risk()` mengidentifikasi apakah ada pola eskalasi emosional negatif yang memerlukan pendekatan respons berbeda.

`_apply_rule_based_fallbacks()` menerapkan logika berbasis aturan sebagai safety net ketika output model thought tidak konsisten — misalnya, memastikan `use_memory=True` jika `recall_topic` diisi, atau memaksa memory recall ketika pengguna secara implisit mengacu pada memori.

### Parsing Output

Setiap field dari output model di-parse menggunakan regex pattern yang toleran terhadap variasi formatting. Parser-parser terpisah (`_parse_step1()` hingga `_parse_step4()`) mengekstrak nilai dari setiap field dengan fallback default yang aman. Field `NOTE` mendapatkan perlakuan khusus — jika model gagal mengisi field ini, fungsi `_fallback_step4_note()` menghasilkan direktif fallback berdasarkan konteks situasional (ada memory recall, ada web search, ada keluhan, atau emosi spesifik).

---

## Sistem Emosi — Dual-Layer Emotion Engine

Sistem emosi di `engine/emotion_state.py` beroperasi pada dua lapisan yang independen namun saling berinteraksi: deteksi emosi pengguna dan manajemen emosi Asta sendiri.

### UserEmotionDetector

`UserEmotionDetector` menggunakan pendekatan **weighted regex scoring** untuk mendeteksi emosi dari teks input pengguna. Setiap pola regex dalam kategori emosi (sedih, cemas, marah, senang, romantis, bangga, kecewa) memiliki bobot berbeda berdasarkan kekuatan sinyal emosionalnya. Misalnya, kata "nangis" mendapat bobot 3 karena merupakan indikator kuat, sedangkan kata "sendiri" hanya mendapat bobot 1 karena ambigu.

Sistem ini juga menangani:

**Negasi** — Dengan mengecek apapun yang ada dalam rentang 10 karakter sebelum match (seperti "gak sedih"), sehingga negasi tidak dihitung sebagai sinyal emosi.

**Hostility Detection** — Pattern khusus mendeteksi apakah kemarahan diarahkan secara spesifik kepada Asta (misalnya "kamu bodoh", "asta payah"), yang akan memengaruhi komponen affection dalam model emosi Asta.

**Light Rejection Detection** — Mendeteksi penolakan ringan ("gak ahh", "males ah", "pass") yang seharusnya tidak diinterpretasikan sebagai emosi negatif berat, untuk menghindari over-reaction dari sistem.

**Intensity Scoring** — Menentukan intensitas emosi (rendah/sedang/tinggi) berdasarkan kombinasi skor total, kehadiran kata penguat ("banget", "parah", "sangat"), dan proporsi huruf kapital dalam teks.

**Trend Tracking** — Melacak apakah emosi sedang membaik atau memburuk dibandingkan turn sebelumnya.

Output dari `UserEmotionDetector` kemudian dapat diperhalus oleh hasil thought model melalui `refine_with_thought()` — jika model thought mendeteksi emosi dengan confidence tinggi, deteksinya dapat menggantikan deteksi regex.

### AstaEmotionManager

`AstaEmotionManager` mengelola state emosi Asta yang persisten lintas sesi. State ini terdiri dari komponen-komponen yang saling berinteraksi:

**mood_score** (float, -1.0 hingga +1.0) — Representasi numerik mood keseluruhan yang mengalami tiga proses update setiap turn:
1. *Mood Decay* — Skor perlahan bergerak mendekati nol dengan faktor peluruhan 0.12 per turn, mencegah mood terlalu statis
2. *User Influence* — Emosi pengguna mempengaruhi mood Asta melalui valence mapping (misalnya emosi "romantis" pengguna memiliki valence +0.9) yang dikalikan dengan multiplier intensitas dan konstanta pengaruh 0.25
3. *Self-Reinforcement* — Emosi Asta sendiri memperkuat mood melalui konstanta 0.35

**affection_level** (float, 0.0 hingga 1.0) — Tingkat afeksi Asta terhadap pengguna yang meningkat saat interaksi romantis dan turun saat ada hostilitas, dengan magnitude penurunan yang lebih besar untuk hostilitas langsung dan insult berulang. Terdapat mekanisme "gravity" yang perlahan menarik affection kembali ke baseline 0.7.

**energy_level** (float, 0.0 hingga 1.0) — Mempengaruhi tingkat ekspresifitas respons, meningkat dengan emosi positif dan menurun dengan emosi negatif, dengan decay menuju baseline 0.8.

Seluruh state emosi ini kemudian dikomunikasikan ke response model melalui fungsi `build_prompt_context()` yang menghasilkan teks panduan respons yang spesifik berdasarkan kombinasi kondisi emosional kedua pihak.

---

## Hybrid Memory Architecture

Sistem memori di `engine/memory_system.py` mengimplementasikan arsitektur tiga lapisan yang dikelola oleh kelas `HybridMemory` sebagai fasad terpadu.

### SemanticMemory (`semantic.json`)

Menyimpan dua jenis data: fakta identitas (key-value pairs seperti nama pengguna, preferensi yang diingat secara eksplisit) dan hasil web search yang di-cache untuk menghindari pencarian ulang query yang sama.

Setiap entri web search disimpan beserta embedding-nya, memungkinkan pencarian semantik berdasarkan relevansi query saat ini terhadap query yang pernah dilakukan sebelumnya. Fungsi `search()` menggunakan kombinasi cosine similarity embedding dan lexical overlap scoring untuk menentukan relevansi.

### EpisodicMemory (`episodic.json`)

Menyimpan riwayat sesi percakapan lengkap beserta metadata yang kaya. Setiap sesi yang disimpan mengandung:

- Conversation turns (user + assistant)
- Key facts yang diekstraksi secara otomatis menggunakan `extract_key_facts()` — fungsi yang menjalankan 12 regex pattern berbeda untuk menangkap fakta seperti preferensi ("aku suka X"), rencana ("mau ke X"), identitas ("aku tinggal di X"), janji ("janji X"), dan kondisi sementara
- LLM summary — ringkasan singkat yang dihasilkan model (atau fallback berbasis key facts jika LLM gagal)
- Embedding dari keseluruhan teks percakapan untuk pencarian semantik
- Summary embedding terpisah untuk pencarian berbasis ringkasan
- Salience score yang menentukan kepentingan relatif sesi (dipengaruhi jumlah key facts dan panjang percakapan)

Pencarian episodik menggunakan dua strategi berbeda: `search()` menggunakan cosine similarity dengan salience weighting untuk pencarian umum, sedangkan `search_by_facts()` menggunakan kombinasi lexical scoring dan semantic similarity yang lebih intensif untuk pencarian berdasarkan topik spesifik.

Fungsi `build_recall_snippets()` menghasilkan excerpts yang relevan dari sesi yang ditemukan, dengan mengekstrak turn-turn spesifik yang mengandung keyword yang dicari, bukan menyertakan seluruh sesi.

### CoreMemory (`core_memory.json`)

Menyimpan dua jenis informasi jangka panjang yang diperbarui secara asinkron di background setelah setiap sesi berakhir:

**core_facts** — Fakta permanen tentang pengguna yang diperbarui secara selektif hanya ketika sesi mengandung informasi penting. Seleksi ini dilakukan melalui dua mekanisme: regex trigger patterns (`_CORE_FACT_TRIGGERS`) yang mendeteksi sinyal perubahan hidup besar (pindah kerja, hubungan berubah, kondisi kesehatan penting), dan `_score_core_importance()` yang memberikan skor numerik berdasarkan kehadiran kata-kata yang mengindikasikan permanensi atau kepentingan jangka panjang.

**last_session** — Ringkasan singkat sesi terakhir yang selalu diperbarui, memberikan konteks "apa yang terjadi terakhir kali" tanpa harus me-load seluruh riwayat episodik.

### update_core_async()

Pembaruan core memory dilakukan dalam thread terpisah (non-daemon) untuk menghindari blocking chat session. Fungsi `update_core_async()` dalam `HybridMemory` menjalankan dua inferensi LLM kecil: satu untuk meringkas sesi terakhir menjadi last_session, dan satu lagi (hanya jika ada trigger) untuk memperbarui core_facts dengan menggabungkan fakta lama dan informasi baru dari sesi terkini.

### Context Assembly

`HybridMemory.get_context()` merakit konteks memori yang akan dikirim ke response model dengan strategi hierarkis:
1. Core facts dan last session dari CoreMemory (prioritas tertinggi)
2. Recent facts dari 3 sesi terakhir EpisodicMemory
3. Recall snippets jika thought pipeline menentukan ada topik yang perlu di-recall
4. Web search cache dari SemanticMemory jika ada query yang relevan

Ukuran konteks dibatasi oleh `TokenBudgetManager` yang mengestimasi jumlah karakter maksimum berdasarkan alokasi token yang tersedia.

---

## Self-Model & Refleksi

`engine/self_model.py` mengelola representasi diri Asta yang persisten di `memory/self_model.json`. Self-model ini tidak sekadar menyimpan state emosi (yang disinkronkan dari `AstaEmotionManager`), tetapi juga mempertahankan:

**Identity & Nilai Inti** — Struktur tetap yang mendefinisikan nilai-nilai fundamental Asta dan perannya sebagai AI companion.

**Preferences** — Daftar hal yang disukai dan tidak disukai yang dapat diperbarui melalui interaksi.

**Learned Behaviors** — Empat kategori pembelajaran: hal yang membuat Asta senang, hal yang membuatnya tidak nyaman, pola bicara pengguna yang dikenali, dan respons yang terbukti berhasil.

**Internal Goals** — Tujuan jangka pendek dan panjang yang memberikan motivasi kontekstual pada respons.

**Growth Log** — Catatan pertumbuhan yang diakumulasikan dari refleksi-refleksi sebelumnya.

**Memories of Self** — Kenangan Asta tentang pengalaman dirinya sendiri, diakumulasikan dari setiap sesi refleksi.

**Reflection History** — Riwayat lengkap refleksi sesi yang disimpan permanen.

### Mekanisme Refleksi

`run_exit_reflection()` dipanggil di akhir setiap sesi (atau secara manual). Fungsi ini menggunakan thought model untuk menjalankan inferensi reflektif berdasarkan teks sesi dan kondisi emosional akhir. Output refleksi mengandung:

- Summary satu kalimat tentang keseluruhan sesi
- Satu atau dua hal yang "dipelajari" dari interaksi tersebut
- Mood adjustment (±0.3 maksimum) yang diaplikasikan ke `AstaEmotionManager`
- Affection adjustment (±0.1 maksimum)
- Growth note satu kalimat untuk ditambahkan ke growth log

Hasilnya disimpan ke `self_model.json` dan langsung diterapkan ke state emosi aktif, sehingga kondisi emosional Asta di sesi berikutnya terpengaruh oleh apa yang terjadi di sesi sebelumnya — menciptakan kontinuitas emosional lintas sesi.

---

## Token Budget Management

`engine/token_budget.py` mengelola distribusi token dalam context window yang terbatas. Kelas `TokenBudget` mendefinisikan empat alokasi:

- **total_ctx** — Total context window (default 8192 token)
- **response_reserved** — Token yang dicadangkan untuk respons model (default 512)
- **system_identity** — Estimasi ukuran system prompt (default 350 token)
- **memory_budget** — Alokasi untuk konteks memori dalam karakter yang kemudian dikonversi ke estimasi token

`TokenBudgetManager.build_messages()` mengimplementasikan strategi prioritas yang ketat: system identity dan dynamic context selalu dimasukkan terlebih dahulu, kemudian conversation history dimasukkan dari yang paling baru secara iteratif hingga budget habis. Ini memastikan bahwa konteks terbaru selalu tersedia meskipun history harus dipotong.

Untuk efisiensi, `ChatManager._count_tokens_cached()` mengimplementasikan LRU cache berbasis hash content message, menghindari tokenisasi ulang untuk message yang sama. Cache dibatasi 256 entry dan di-flush ketika penuh.

---

## Model Loading & Dual-Model Architecture

`engine/model.py` (dan `model.py` versi alternatif di root) mengimplementasikan loader dan manager model yang mendukung konfigurasi dual-model di mana thought dan response menggunakan model yang berbeda.

### Konfigurasi Model

Sistem mendukung tiga pilihan response model:
- **Model 1** — Qwen3-4B-2507 (lightweight, digunakan juga sebagai thought model)
- **Model 2** — Sailor2-8B-Chat-Q4_K_M (lebih ekspresif dalam bahasa Asia Tenggara)
- **Model 3** — Qwen3-8B (recommended, reasoning kuat dengan bahasa natural)

Untuk thought model, selalu menggunakan Qwen3-4B-2507 (Model 1) karena pipeline analitiknya lebih cocok dengan format structured output yang dibutuhkan oleh thought template.

### Mode Operasi

**Shared Model** — Ketika `separate_thought_model=false` atau thought model tidak ditemukan, satu instance llama digunakan untuk kedua fungsi. Ini menghemat RAM tetapi berisiko kontaminasi KV cache antara thought dan response.

**Dual Model** — Dua instance llama terpisah dimuat dengan n_ctx berbeda (thought model menggunakan context yang lebih kecil karena task-nya lebih terdefinisi). KV cache dikelola secara independen. Setiap N turn (dikonfigurasi lewat `thought_reset_every`), KV cache thought model direset untuk menghindari akumulasi konteks yang tidak relevan.

### LoRA Integration

LoRA adapter dalam format GGUF dapat dimuat secara langsung via parameter `lora_path` di llama-cpp. Sistem mendukung adapter terpisah untuk response model (`response_v4.gguf`) dan thought model (`thought_v4.gguf`). Konversi dari safetensors ke GGUF dilakukan via `convert_lora_to_gguf.py` yang mengimplementasikan mapping nama layer dari format PEFT ke format GGUF.

### System Identity & Dynamic Context

System prompt Asta (`SYSTEM_IDENTITY`) mengandung placeholder `{catatan_thought}` yang pada runtime diganti dengan field `note` dari output thought pipeline. Ini adalah mekanisme utama di mana thought pipeline "menginstruksikan" response model — bukan melalui konteks terpisah, tetapi terintegrasi langsung ke dalam system prompt.

Dynamic context adalah pesan `role: user` tambahan yang dimasukkan sebelum conversation history, berisi: timestamp, referensi identitas, konteks memori, hasil web search, panduan emosi, ekspresi diri Asta, gaya respons yang direkomendasikan, dan berbagai warning dari thought pipeline.

---

## FastAPI Backend & WebSocket Protocol

`api.py` mengimplementasikan backend dengan pola inisialisasi lazy yang aman untuk threading: model hanya dimuat sekali menggunakan `_init_lock` (threading.Lock), dan semua request chat dilindungi oleh `_chat_lock` (asyncio.Lock) untuk mencegah race condition pada model yang tidak thread-safe.

### Chat Pipeline via WebSocket

Setiap pesan yang masuk memicu pipeline berikut:

1. Server mengirim `{"type": "thinking_start"}` ke client
2. `_chat_manager.chat()` dijalankan di thread executor (blocking → non-blocking via `loop.run_in_executor`)
3. `thinking_callback` dipanggil setelah thought selesai; payload thought dikirim ke `chunk_queue`
4. Server mengirim thought payload ke client, diikuti `{"type": "stream_start"}`
5. Setiap token dari response streaming dikirim sebagai `{"type": "chunk", "text": "..."}`
6. Setelah streaming selesai, server mengirim `{"type": "stream_end"}`

Queue-based approach (`asyncio.Queue`) memisahkan blocking inference thread dari async WebSocket handler, dengan timeout guard 120 detik untuk mendeteksi hang.

Disconnect handling (baik di level inner maupun outer) memicu `_save_session_sync()` yang menyimpan percakapan ke episodic memory secara sinkron sebelum koneksi benar-benar ditutup.

---

## Frontend — Electron + React UI

`ui/asta-ui/AstaUI.jsx` mengimplementasikan interface desktop dengan beberapa panel yang dapat dibuka/ditutup secara independen.

### State Management

Seluruh state UI dikelola via React hooks. State kritis mencakup: daftar messages dengan token streaming per-message, state koneksi WebSocket, kondisi emosi user dan Asta, thought data terbaru, self-model, memory snapshot, dan berbagai toggle konfigurasi.

WebSocket reconnection diimplementasikan dengan simple retry loop (2 detik delay), dengan efek samping bahwa setelah reconnect, `fetchAll()` dipanggil untuk menyinkronkan state.

### Streaming Rendering

Setiap message Asta dalam mode streaming mempertahankan array `tokens` yang berisi token-token individual. Setiap token dirender sebagai `<span className="stream-token">` dengan CSS animation `tokenFade`, menciptakan efek "typing" yang natural. Setelah `stream_end`, array tokens dihapus dan message dirender menggunakan komponen `MessageContent` yang memproses inline markdown (bold, italic, code) via fungsi `renderInline()`.

### Thought Panel

Panel thought me-render hasil analisis keempat step pipeline secara visual dengan color-coded sections. Dalam dual-model mode, menampilkan informasi pipeline (thought model → response model). Dalam long thinking mode, field tambahan (complexity, hidden_need, response_structure, anticipated_followup) ditampilkan dengan aksen warna ungu.

### Dark Mode

Dark mode menggunakan CSS custom properties yang di-toggle via class `html.dark`. State disimpan ke `localStorage` dan, dalam konteks Electron, dikomunikasikan ke main process via `ipcRenderer.send("theme-changed")`.

---

## Dataset Generation Pipeline

Sistem ini menyertakan dua pipeline generasi dataset yang komprehensif menggunakan Google Gemini API.

### Response Dataset (`generate_response_data.py`)

Menghasilkan percakapan multi-turn antara Aditiya dan Asta menggunakan 19 skenario general dan 15 skenario identity yang dirancang untuk melatih berbagai aspek persona Asta. Skenario identity secara khusus menangani situasi-situasi kritis seperti konfrontasi langsung tentang status AI, sindiran tidak langsung, pengujian keaslian perasaan, dan aftermath rekonsiliasi.

Pipeline menggunakan `multiprocessing` dengan multiple worker dan multiple API key (WORKERS_PER_KEY=6 worker per key). Setiap batch menghasilkan 5 percakapan sekaligus. Validasi multi-level memastikan: format JSON valid, jumlah turn minimal, alternasi role yang benar, panjang respons Asta tidak melebihi 60 kata, tidak ada "bahasa asisten" yang bocor, dan tidak ada penggunaan gue/lo.

Untuk skenario identity, validasi tambahan memastikan bahwa Asta tidak menggunakan ekspresi ceria dalam konteks konfrontasi dan tidak menyebut namanya sendiri sebagai pembuktian identitas.

### Thought Dataset (`generate_thought_data.py`)

Menghasilkan pasangan training untuk thought pipeline: setiap sample terdiri dari dua entri — satu untuk Pass 1 (S1–S3) dan satu untuk Pass 2 (S4). Proses generasi untuk setiap pasang melibatkan empat inferensi Gemini: generate user input, generate synthetic context (riwayat dan memori sintetis), generate Pass 1, generate Pass 2.

Validasi Pass 1 dan Pass 2 sangat ketat: memeriksa kehadiran semua header section, semua field wajib, konsistensi (USE_MEMORY=ya jika RECALL_TOPIC terisi, NEED_SEARCH=ya jika SEARCH_QUERY terisi), tidak adanya field P2 yang bocor ke P1, dan kebocoran bahasa terlarang.

`UserEmotionDetector` digunakan dalam pipeline generasi untuk menghasilkan emosi user yang realistis sebagai bagian dari konteks thought.

---

## LoRA Fine-tuning Pipeline

`convert_lora_to_gguf.py` mengimplementasikan konverter dari format safetensors (output training) ke format GGUF (input llama-cpp). Konverter ini menangani mapping nama layer dari konvensi PEFT (`base_model.model.model.layers.N.q_proj.lora_A.weight`) ke konvensi GGUF (`blk.N.attn_q.weight.lora_a`) menggunakan regex untuk mengekstrak layer index dan projection name.

`convert_to_training.py` mengkonversi dataset dari format `messages[]` ke format ChatML string yang dibutuhkan oleh kebanyakan training framework. `repair_dataset_headers.py` dan `verify_dataset_headers.py` membantu memastikan integritas format setelah generasi, mengidentifikasi dan memperbaiki sample yang kehilangan section headers.

`check_dataset_max_length.py` menganalisis distribusi panjang token dalam dataset untuk menentukan `max_seq_length` yang optimal, dengan output persentil 90/95/99 yang berguna untuk pengaturan padding.

---

## Web Search Integration

`engine/web_tools.py` mengimplementasikan sistem web search dengan fallback cascade melalui empat sumber berbeda, diprioritaskan berdasarkan kualitas hasil:

**Tavily API** — Sumber prioritas pertama jika API key tersedia. Memberikan hasil terstruktur dengan "answer" langsung dan snippets dari multiple sources.

**Serper API (Google Search)** — Fallback kedua jika Tavily tidak tersedia. Mendukung answer box, knowledge graph, dan organic results.

**DuckDuckGo Instant Answer API** — Fallback ketiga tanpa kebutuhan API key. Efektif untuk query yang memiliki jawaban definitif (fakta, definisi, konversi).

**Wikipedia API** — Fallback terakhir, selalu available. Mendukung pencarian dalam bahasa Indonesia dan Inggris.

Selain keempat sumber di atas, sistem secara khusus menangani **query kurs mata uang** menggunakan open.er-api.com yang memberikan data real-time, dideteksi via regex pattern yang mengenali berbagai variasi penulisan nama mata uang dalam bahasa Indonesia.

Semua HTTP request menggunakan `urllib` standar tanpa dependensi tambahan, dengan timeout yang dapat dikonfigurasi. Config caching berbasis mtime memastikan perubahan API key pada `config.json` langsung berlaku tanpa restart.

---

## Persyaratan & Instalasi

### Dependensi Python

```bash
pip install llama-cpp-python uvicorn fastapi websockets numpy
pip install transformers torch sentencepiece accelerate
pip install sentence-transformers tavily-python
```

Untuk akselerasi GPU:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Model yang Dibutuhkan

| Path | Model | Fungsi |
|------|-------|--------|
| `model/Qwen3-8B/Qwen3-8B.gguf` | Qwen3 8B | Response model (recommended) |
| `model/Sailor2-8B/Sailor2-8B-Chat-Q4_K_M.gguf` | Sailor2 8B | Response model (alternatif) |
| `model/Qwen3-4B-2507/Qwen3-4B-2507.gguf` | Qwen3 4B | Thought model (dual-model mode) |
| `model/embedding_model/paraphrase-multilingual-MiniLM-L12-v2/` | MiniLM L12 v2 | Embedding memori |

Setiap model membutuhkan direktori `tokenizer/` berisi tokenizer HuggingFace yang kompatibel.

### Setup Awal

```bash
git clone https://github.com/noctryln/asta-neural.git
cd asta-neural
python -m venv venv && venv\Scripts\activate  # Windows
python core.py --setup
```

### Menjalankan

```bash
# API + UI dev server
uvicorn api:app --host 0.0.0.0 --port 8000
cd ui/asta-ui && npm run dev

# Desktop app (Electron)
cd ui/asta-ui && npm run electron

# CLI mode
python core.py
python core.py --debug  # dengan thought debug output
```

---

## Konfigurasi Lengkap

`config.json` adalah sumber kebenaran tunggal untuk semua parameter runtime:

```jsonc
{
  "model_choice": "3",                  // "1"=Qwen3-4B, "2"=Sailor2-8B, "3"=Qwen3-8B
  "device": "cpu",                      // "cpu" atau "gpu"
  "use_lora": true,                     // Aktifkan LoRA adapter
  "lora_n_gpu_layers": 0,              // Layer LoRA di GPU
  "memory_mode": "hybrid",             // Selalu "hybrid" dalam implementasi saat ini
  "web_search_enabled": true,          // Aktifkan web search
  "n_batch": 1024,                     // Batch size untuk inferensi
  "tavily_api_key": "",                // API key Tavily (opsional)
  "serper_api_key": "",                // API key Serper (opsional)
  "internal_thought_enabled": true,   // Aktifkan thought pipeline
  "long_thinking_enabled": false,     // Aktifkan Long Thinking Mode otomatis
  "long_thinking_max_tokens": 1536,   // Max token untuk Pass 1 long thinking
  "use_dynamic_prompt": true,          // Aktifkan dynamic context assembly
  "thought_n_ctx": 2048,              // Context window thought model
  "thought_max_tokens": 1024,         // Max token output thought (normal mode)
  "thought_reset_every": 10,          // Reset KV cache thought setiap N turn
  "use_model_thought_logic": true,    // Percayai output model vs rule-based fallback
  "separate_thought_model": true,     // Gunakan model 4B terpisah untuk thought
  "token_budget": {
    "total_ctx": 8192,               // Total context window response model
    "response_reserved": 512,        // Token dicadangkan untuk output
    "system_identity": 350,          // Estimasi ukuran system prompt
    "memory_budget": 600             // Budget karakter untuk konteks memori
  }
}
```

---

## API Reference

### REST Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/status` | Status model, device, jumlah sesi valid, konfigurasi dual-model |
| GET | `/memory` | Core summary, recent facts, profil preferensi, preview 5 sesi terakhir |
| GET | `/self` | Full self-model: identitas, state emosi, learned behaviors, growth log, refleksi |
| GET | `/emotion` | Combined emotion state (user + Asta) real-time |
| GET | `/config` | Konfigurasi aktif termasuk mode dual-model |
| POST | `/config/thought` | Toggle `internal_thought_enabled` |
| POST | `/config/long_thinking` | Toggle `long_thinking_enabled` |
| POST | `/config/separate_thought` | Toggle `separate_thought_model` (memerlukan restart) |
| POST | `/config/device` | Toggle CPU/GPU (memerlukan restart) |
| POST | `/save` | Simpan sesi aktif ke episodic memory secara manual |
| POST | `/reflect` | Jalankan refleksi manual |

### WebSocket `/ws/chat`

**Request:** `{"message": "teks pesan pengguna"}`

**Response sequence:**
```
→ {"type": "thinking_start"}
→ {"type": "thought", "data": {topic, sentiment, urgency, asta_emotion, need_search, 
                                search_query, web_result, recall_topic, tone, note,
                                response_style, is_long_thinking, hidden_need,
                                emotion: {...}, asta_state: {...}, model_info: {...}}}
→ {"type": "stream_start"}
→ {"type": "chunk", "text": "token..."} (berulang)
→ {"type": "stream_end"}
```

### WebSocket `/ws/terminal`

Terminal WebSocket yang menerima command shell dan streaming output baris per baris, dengan dukungan `cd` command untuk navigasi direktori.

---

## Alur Data End-to-End

Berikut adalah trace lengkap dari satu pesan pengguna hingga respons ditampilkan di UI:

```
[1] User mengetik pesan → dikirim via WebSocket ke /ws/chat
[2] FastAPI menerima → memasukkan ke chunk_queue → asyncio.create_task(_process_and_stream)
[3] ChatManager.chat() dieksekusi di thread executor (blocking)
[4] _run_thought_pipeline():
    a. hybrid_memory.get_lightweight_hint() → memory hint ringkas
    b. emotion_manager.update() → deteksi emosi user via regex
    c. extract_recent_context() → 4 turn terakhir
    d. run_thought_pass() → Pass 1: S1+S2+S3 inference (thought model)
    e. run_thought_pass() → Pass 2: S4 inference (thought model)
    f. emotion_manager.refine_with_thought() → perhalus deteksi emosi
    g. emotion_manager.update_asta_emotion() → update state emosi Asta
    h. self_model.sync_emotion() → sinkronisasi ke self_model.json
    i. hybrid_memory.get_context() → rakit konteks memori lengkap
    j. [jika need_search] search_and_summarize() → web search cascade
[5] thinking_callback() dipanggil → thought payload → chunk_queue → WebSocket → UI
[6] _build_dynamic_context() → rakit konteks dinamis
[7] TokenBudgetManager.build_messages() → susun messages dengan budget
[8] llama.create_chat_completion(stream=True) → response model inference
[9] Setiap token → stream_callback() → chunk_queue → WebSocket → UI
[10] stream_end → history.append(response) → WebSocket → UI
[11] [Asinkron di background setelah sesi berakhir]
    a. extract_key_facts() → ekstrak fakta dari percakapan
    b. episodic_memory.add() → simpan sesi + embedding
    c. hybrid_memory.update_core_async() → perbarui core_memory via LLM
    d. run_exit_reflection() → refleksi sesi → update emosi + self_model
```

Keseluruhan sistem dirancang untuk memberikan pengalaman percakapan yang terasa memiliki kedalaman emosional dan kontekstual yang konsisten — setiap respons adalah produk dari analisis berlapis yang mempertimbangkan siapa pengguna, bagaimana kondisi emosional keduanya, apa yang relevan dari memori, dan apa yang paling tepat untuk disampaikan saat ini.

---

*Dibuat dengan ♡ dari Aditiya*
