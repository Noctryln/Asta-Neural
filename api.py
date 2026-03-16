"""
api.py — FastAPI + WebSocket backend for Asta AI
Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Lazy-load heavy Asta modules so the API can start faster ──────────────────
_chat_manager = None
_hybrid_memory = None
_init_lock = threading.Lock()
_initialized = False


def _initialize():
    global _chat_manager, _hybrid_memory, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        from config import load_config
        from engine.model import load_model
        from engine.memory import get_hybrid_memory, get_identity, remember_identity

        cfg = load_config()
        chat_manager = load_model(cfg)
        hybrid_mem = get_hybrid_memory()
        chat_manager.hybrid_memory = hybrid_mem

        user_name = get_identity("nama_user") or "Aditiya"
        chat_manager.system_identity += f"\n- Nama pengguna: {user_name}."
        chat_manager._user_name_cache = user_name

        _chat_manager = chat_manager
        _hybrid_memory = hybrid_mem
        _initialized = True


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Asta AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _initialize)


# ── REST: Memory & Status ─────────────────────────────────────────────────────

@app.get("/status")
async def status():
    if not _initialized:
        return {"ready": False}
    import numpy as np
    ep_count = len([
        s for s in _hybrid_memory.episodic.data
        if not np.allclose(np.array(s.get("embedding", [0])[:5]), 0.0)
    ])
    return {
        "ready": True,
        "model": _chat_manager.cfg.get("model_choice", "?"),
        "user_name": _chat_manager._user_name_cache,
        "episodic_sessions": ep_count,
    }


@app.get("/memory")
async def get_memory():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    core_text = _hybrid_memory.core.get_context_text()
    recent_facts = _hybrid_memory.episodic.get_recent_facts_text(n_sessions=3, max_facts=10)
    profile = _hybrid_memory.core.get_profile()
    sessions = _hybrid_memory.episodic.get_last_n(5)
    session_previews = []
    for s in sessions:
        conv = s.get("conversation", [])
        first_user = next((m["content"] for m in conv if m["role"] == "user"), "")
        session_previews.append({
            "timestamp": s.get("timestamp", ""),
            "preview": first_user[:80],
            "facts": len(s.get("key_facts", [])),
        })
    return {
        "core": core_text,
        "recent_facts": recent_facts,
        "profile": profile,
        "sessions": session_previews,
    }


@app.get("/emotion")
async def get_emotion():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    return _chat_manager.emotion_manager.get_state()


@app.get("/config")
async def get_config():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    return {
        "internal_thought_enabled": _chat_manager.cfg.get("internal_thought_enabled", True),
        "web_search_enabled": _chat_manager.cfg.get("web_search_enabled", True),
    }


@app.post("/config/thought")
async def toggle_thought():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    from config import save_config
    current = _chat_manager.cfg.get("internal_thought_enabled", True)
    _chat_manager.cfg["internal_thought_enabled"] = not current
    save_config(_chat_manager.cfg)
    return {"internal_thought_enabled": _chat_manager.cfg["internal_thought_enabled"]}



async def save_session():
    if not _initialized:
        return JSONResponse({"error": "not initialized"}, status_code=503)
    from engine.memory import add_episodic
    conv = [
        {"role": m["role"], "content": m["content"]}
        for m in _chat_manager.conversation_history
        if m["content"]
    ]
    if conv:
        _hybrid_memory.extract_and_save_preferences(conv)
        add_episodic(conv)
        session_text = _chat_manager.get_session_text()
        if session_text:
            _hybrid_memory.update_core_async(
                llm_callable=_chat_manager.llama.create_completion,
                current_session_text=session_text,
            )
    return {"saved": len(conv)}


# ── WebSocket: Streaming Chat ─────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                user_input = data.get("message", "").strip()
                if not user_input:
                    continue
            except json.JSONDecodeError:
                user_input = raw.strip()

            if not _initialized:
                await websocket.send_text(json.dumps({
                    "type": "error", "text": "Model belum siap."
                }))
                continue

            # Send thought/status before streaming
            await websocket.send_text(json.dumps({"type": "thinking_start"}))

            thought_holder = {}
            emotion_holder = {}
            # Queue untuk streaming real-time dari thread ke event loop
            chunk_queue = asyncio.Queue()

            def _run_chat():
                import datetime
                from engine.thought import (
                    run_thought_pass, build_augmented_system,
                    extract_recent_context,
                )
                from engine.web_tools import search_and_summarize

                cm = _chat_manager
                now = datetime.datetime.now()
                timestamp_str = now.strftime("%A, %d %B %Y, pukul %H:%M WIB")
                system_with_time = cm.system_identity + f"\n- Waktu sekarang: {timestamp_str}."

                recent_ctx = extract_recent_context(cm.conversation_history, n=2)
                emotion_state = cm.emotion_manager.update(user_input, recent_context=recent_ctx)

                memory_ctx = cm._get_memory_context(query=user_input, recall_topic="")

                thought = {
                    "need_search": False, "search_query": "",
                    "recall_topic": "", "tone": "romantic", "note": "", "raw": ""
                }
                if cm.cfg.get("internal_thought_enabled", True):
                    thought = run_thought_pass(
                        llm=cm.llama,
                        user_input=user_input,
                        memory_context="",
                        recent_context=recent_ctx,
                        web_search_enabled=cm.cfg.get("web_search_enabled", True),
                        max_tokens=50,
                        user_name=cm._user_name_cache,
                        emotion_state=(
                            f"emosi={emotion_state['user_emotion']}; "
                            f"intensitas={emotion_state['intensity']}; "
                            f"tren={emotion_state['trend']}"
                        ),
                    )
                    emotion_state = cm.emotion_manager.refine_with_thought(thought)

                thought_holder["thought"] = thought
                emotion_holder["emotion"] = emotion_state

                emotion_guidance = cm.emotion_manager.build_prompt_context()

                web_result = ""
                if (
                    cm.cfg.get("web_search_enabled", True)
                    and thought["need_search"]
                    and thought.get("search_query")
                ):
                    web_result = search_and_summarize(
                        thought["search_query"], max_results=2, timeout=5
                    )
                    if not web_result:
                        web_result = "[INFO] Web search gagal."

                augmented_system = build_augmented_system(
                    base_system=system_with_time,
                    thought=thought,
                    memory_context=memory_ctx,
                    web_result=web_result,
                    emotion_guidance=emotion_guidance,
                )

                system_msg = {"role": "system", "content": augmented_system}
                cm.conversation_history.append({"role": "user", "content": user_input})

                messages_to_send, _ = cm.budget_manager.build_messages(
                    system_identity=system_msg,
                    memory_messages=[],
                    conversation_history=cm.conversation_history,
                )

                # Kirim sinyal: thought selesai, mulai stream
                loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "thought_ready"})

                response_stream = cm.llama.create_chat_completion(
                    messages=messages_to_send,
                    max_tokens=128,
                    temperature=0.7,
                    top_p=0.85,
                    top_k=60,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    stream=True,
                )

                full_response = ""
                for chunk in response_stream:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        text = delta["content"]
                        full_response += text
                        # Push chunk langsung ke event loop — ini yang bikin streaming real
                        loop.call_soon_threadsafe(
                            chunk_queue.put_nowait, {"type": "chunk", "text": text}
                        )

                cm.conversation_history.append({"role": "assistant", "content": full_response})
                # Sinyal selesai
                loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "done"})

            # Jalankan inference di thread terpisah
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = loop.run_in_executor(executor, _run_chat)

            # Baca dari queue dan kirim ke WebSocket secara real-time
            stream_started = False
            while True:
                item = await chunk_queue.get()

                if item["type"] == "thought_ready":
                    # Kirim thought metadata
                    thought = thought_holder.get("thought", {})
                    emotion = emotion_holder.get("emotion", {})
                    await websocket.send_text(json.dumps({
                        "type": "thought",
                        "data": {
                            "need_search": thought.get("need_search", False),
                            "search_query": thought.get("search_query", ""),
                            "recall_topic": thought.get("recall_topic", ""),
                            "tone": thought.get("tone", ""),
                            "note": thought.get("note", ""),
                            "emotion": emotion,
                        }
                    }))
                    await websocket.send_text(json.dumps({"type": "stream_start"}))
                    stream_started = True

                elif item["type"] == "chunk":
                    await websocket.send_text(json.dumps({"type": "chunk", "text": item["text"]}))

                elif item["type"] == "done":
                    await websocket.send_text(json.dumps({"type": "stream_end"}))
                    break

            await future  # pastikan thread selesai

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "text": str(e)}))
        except Exception:
            pass