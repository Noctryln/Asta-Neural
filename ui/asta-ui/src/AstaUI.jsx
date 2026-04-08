import { useState, useEffect, useRef, useCallback } from "react";

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#f7f6f4; --surface:#fff; --surface2:#f0eeeb; --border:#e5e2dd;
  --text:#1a1816; --muted:#8c8680; --accent:#d4a96a; --asta:#1a1816;
  --tag-bg:#edeae5; --r:25px; --rs:8px;
  --shadow:0 2px 16px rgba(0,0,0,0.07);
  --font:'Sora',sans-serif; --mono:'JetBrains Mono',monospace;
  --ease:0.22s cubic-bezier(0.4,0,0.2,1);
  --blue:#7a9ec7; --green:#6ab87a; --rose:#c97b8a; --purple:#9b7ac9;
}
html.dark {
  --bg:#141210; --surface:#1e1b18; --surface2:#252018; --border:#2e2a24;
  --text:#e8e0d5; --muted:#6b6560; --accent:#d4a96a; --asta:#c8a882;
  --tag-bg:#2a2520; --shadow:0 2px 16px rgba(0,0,0,0.35);
}
html,body{height:100%;width:100%;overflow:hidden;font-family:var(--font);background:var(--bg);color:var(--text);transition:background .3s,color .3s;}
#root{height:100%;width:100%;display:flex;flex-direction:column;}
::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:99px}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideR{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}
@keyframes slideL{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pullOut{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes popIn{from{opacity:0;transform:scale(.9) translateX(-10px)}to{opacity:1;transform:scale(1) translateX(0)}}
@keyframes tokenFade{from{opacity:0}to{opacity:1}}
@keyframes longThinkPulse{0%,100%{box-shadow:0 0 0 0 rgba(154,122,201,0.4)}50%{box-shadow:0 0 0 6px rgba(154,122,201,0)}}
.stream-token{animation:tokenFade .38s ease forwards}
.note-bubble{animation:popIn .3s cubic-bezier(0.4,0,0.2,1) forwards;transform-origin:left center;}
.note-bubble::after{content:'';position:absolute;left:-7px;top:14px;width:12px;height:12px;background:var(--surface);border-left:1.5px solid var(--green);border-bottom:1.5px solid var(--green);transform:rotate(45deg);z-index:1}

.dm-toggle{
  position:relative;width:40px;height:22px;
  background:var(--border);border-radius:99px;
  cursor:pointer;border:none;
  transition:background .25s;flex-shrink:0;
}
.dm-toggle.is-on { background:var(--accent); }
.dm-toggle.is-on-purple { background:var(--purple); }
.dm-toggle::after{
  content:'';position:absolute;top:3px;left:3px;
  width:16px;height:16px;border-radius:50%;background:white;
  transition:transform .25s cubic-bezier(0.4,0,0.2,1);
}
.dm-toggle.is-on::after,
.dm-toggle.is-on-purple::after { transform:translateX(18px); }

.bar-fill{transition:width .6s ease}
.hide-scrollbar::-webkit-scrollbar{display:none}
.hide-scrollbar{-ms-overflow-style:none;scrollbar-width:none}
.long-think-badge{animation:longThinkPulse 2s ease-in-out infinite}

/* FIX #4: Markdown styles inside chat bubble */
.msg-body { line-height: 1.65; }
.msg-body strong { font-weight: 600; }
.msg-body em { font-style: italic; }
.msg-body code {
  font-family: var(--mono);
  font-size: 0.85em;
  background: rgba(0,0,0,0.08);
  border-radius: 4px;
  padding: 1px 5px;
}
html.dark .msg-body code { background: rgba(255,255,255,0.1); }
.msg-body .msg-line { display: block; min-height: 1em; }
.msg-body .msg-line + .msg-line { margin-top: 4px; }
`;

const WS_URL  = "ws://localhost:8000/ws/chat";
const API_URL = "http://localhost:8000";

const EMO_MAP = {
  netral:           { emoji:"*",   color:"#8c8680", label:"Netral"         },
  senang:           { emoji:"✦",  color:"#d4a96a", label:"Senang"         },
  romantis:         { emoji:"♡",  color:"#c97b8a", label:"Romantis"       },
  sedih:            { emoji:"·",   color:"#7a9ec7", label:"Sedih"          },
  cemas:            { emoji:"~",   color:"#b07ab0", label:"Cemas"          },
  marah:            { emoji:"!",   color:"#c07060", label:"Marah"          },
  rindu:            { emoji:"◦",   color:"#a07ab0", label:"Rindu"          },
  bangga:           { emoji:"★",  color:"#d4a96a", label:"Bangga"         },
  kecewa:           { emoji:"…",   color:"#9ab0c7", label:"Kecewa"         },
  "sangat senang":  { emoji:"✦✦", color:"#d4a96a", label:"Sangat Senang"  },
  "sedikit senang": { emoji:"·",   color:"#c8b87a", label:"Sedikit Senang" },
  murung:           { emoji:"·",   color:"#7a9ec7", label:"Murung"         },
  "sangat murung":  { emoji:"··",  color:"#6080a7", label:"Sangat Murung"  },
};
const getEmo = (key) => EMO_MAP[key] || EMO_MAP.netral;

if (!document.getElementById("asta-css")) {
  const t = document.createElement("style");
  t.id = "asta-css"; t.textContent = CSS;
  document.head.appendChild(t);
}

// ── FIX #4: Inline markdown renderer ─────────────────────────────────────────
// Renders **bold**, *italic*, `code` within a single line segment.
function renderInline(text) {
  const parts = [];
  // Split on **bold**, *italic*, `code`
  const re = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[0].startsWith("**"))      parts.push(<strong key={m.index}>{m[2]}</strong>);
    else if (m[0].startsWith("*"))  parts.push(<em key={m.index}>{m[3]}</em>);
    else if (m[0].startsWith("`"))  parts.push(<code key={m.index}>{m[4]}</code>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// FIX #4: Full message renderer — splits by newline, renders each line with inline markdown.
// Also strips leading "Asta:" prefix that might slip through.
function MessageContent({ content }) {
  // Strip residual "Asta:" / "**Asta:**" prefix
  const clean = content.replace(/^\*{0,2}Asta\*{0,2}\s*:\s*/i, "").trimStart();
  const lines = clean.split("\n");
  return (
    <div className="msg-body">
      {lines.map((line, i) => (
        <span key={i} className="msg-line">
          {renderInline(line)}
        </span>
      ))}
    </div>
  );
}

export default function AstaUI() {
  const [messages,        setMessages]        = useState([]);
  const [input,           setInput]           = useState("");
  const [connected,       setConnected]       = useState(false);
  const [thinking,        setThinking]        = useState(false);
  const [streaming,       setStreaming]       = useState(false);
  const [userEmotion,     setUserEmotion]     = useState({ user_emotion:"netral", intensity:"rendah", trend:"stabil" });
  const [astaEmotion,     setAstaEmotion]     = useState({ current_emotion:"netral", mood:"netral", mood_score:0, affection_level:0.7, energy_level:0.8 });
  const [thought,         setThought]         = useState(null);
  const [selfModel,       setSelfModel]       = useState(null);
  const [memory,          setMemory]          = useState(null);
  const [panel,           setPanel]           = useState(null);
  const [sysStats,        setSysStats]        = useState({ cpu:0, ram:0, disk:0 });
  const [noteVisible,     setNoteVisible]     = useState(false);
  const [serverReady,     setServerReady]     = useState(false);

  const [thoughtEnabled,  setThoughtEnabled]  = useState(true);
  const [separateThought, setSeparateThought] = useState(false);
  const [longThinking,    setLongThinking]    = useState(false);
  const [device,          setDevice]          = useState("cpu");
  const [modelInfo,       setModelInfo]       = useState({ dual_model:false, thought_model:"?", response_model:"?" });
  const [darkMode,        setDarkMode]        = useState(() => localStorage.getItem("asta-dark") === "1");

  const wsRef        = useRef(null);
  const bottomRef    = useRef(null);
  const bufRef       = useRef("");
  const msgIdRef     = useRef(0);
  const tokIdRef     = useRef(0);
  const thoughtRef   = useRef(null);
  const mainInputRef = useRef(null);

  const handleTerminalMessage = useCallback((msg) => {
    if (msg.type === "stats") setSysStats(msg.data);
  }, []);

  useEffect(() => {
    if (connected && !thinking && !streaming && panel !== "terminal")
      mainInputRef.current?.focus();
  }, [connected, thinking, streaming, panel]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem("asta-dark", darkMode ? "1" : "0");
    if (window.require) {
      const { ipcRenderer } = window.require("electron");
      ipcRenderer.send("theme-changed", darkMode ? "dark" : "light");
    }
  }, [darkMode]);

  const sanitize    = t => t ? t.replace(/\uFFFD/g,"").replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g,"") : "";
  const scrollBottom = useCallback(() => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior:"smooth" }), 60);
  }, []);

  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const d = await (await fetch(`${API_URL}/status`)).json();
        if (d.ready) {
          setServerReady(true);
          setModelInfo({ dual_model:d.dual_model||false, thought_model:d.thought_model||"?", response_model:d.response_model||"?" });
          clearInterval(iv);
        }
      } catch(_) {}
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    if (!serverReady) return;
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
      ws.onerror = () => ws.close();
      ws.onmessage = e => {
        const msg = JSON.parse(e.data);
        if (msg.type === "thinking_start") {
          setThinking(true);
        } else if (msg.type === "thought") {
          setThinking(false);
          const td = msg.data;
          setThought(td);
          thoughtRef.current = td;
          if (td.emotion)    setUserEmotion(td.emotion);
          if (td.asta_state) setAstaEmotion(td.asta_state);
          if (td.model_info) setModelInfo(td.model_info);
        } else if (msg.type === "stream_start") {
          bufRef.current = ""; tokIdRef.current = 0;
          setStreaming(true);
          const id = ++msgIdRef.current;
          const td = thoughtRef.current || {};
          const hasWeb = Boolean(td.need_search && td.search_query);
          setMessages(p => [...p, {
            id, role:"assistant", content:"", tokens:[],
            isLong: td.is_long_thinking || false,
            webSearch: hasWeb ? { query:td.search_query||"Web search", result:td.web_result||"Belum ada hasil." } : null,
          }]);
        } else if (msg.type === "chunk") {
          const clean = sanitize(msg.text);
          if (!clean) return;
          bufRef.current += clean;
          const buf = bufRef.current;
          const tid = ++tokIdRef.current;
          setMessages(p => p.map(m =>
            m.id === msgIdRef.current
              ? { ...m, content:buf, tokens:[...(m.tokens||[]), {id:tid, text:clean}] }
              : m
          ));
          scrollBottom();
        } else if (msg.type === "stream_end") {
          setStreaming(false);
          setMessages(p => p.map(m => m.id===msgIdRef.current ? {...m, tokens:null} : m));
          scrollBottom();
          fetchAll();
        } else if (msg.type === "error") {
          setThinking(false); setStreaming(false);
          setMessages(p => [...p, { id:++msgIdRef.current, role:"assistant", content:`⚠ ${msg.text}`, tokens:null }]);
        }
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, [serverReady]);

  const fetchAll = useCallback(async () => {
    try { setMemory(await (await fetch(`${API_URL}/memory`)).json()); } catch(_) {}
    try { setSelfModel(await (await fetch(`${API_URL}/self`)).json()); } catch(_) {}
  }, []);

  useEffect(() => {
    if (!serverReady) return;
    fetchAll();
    fetch(`${API_URL}/config`).then(r=>r.json()).then(d => {
      setThoughtEnabled(d.internal_thought_enabled   ?? true);
      setSeparateThought(d.separate_thought_model    ?? false);
      setLongThinking(d.long_thinking_enabled        ?? false);
      setDevice(d.device || "cpu");
      setModelInfo({ dual_model:d.dual_model||false, thought_model:d.thought_model||"?", response_model:d.response_model||"?" });
    }).catch(_=>{});
  }, [serverReady]);

  useEffect(() => {
    if (thought?.note) {
      setNoteVisible(true);
      const t = setTimeout(() => setNoteVisible(false), 6000);
      return () => clearTimeout(t);
    }
  }, [thought]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || !connected || thinking || streaming) return;
    setMessages(p => [...p, { id:++msgIdRef.current, role:"user", content:text }]);
    setInput(""); scrollBottom();
    wsRef.current?.send(JSON.stringify({ message: text }));
  }, [input, connected, thinking, streaming, scrollBottom]);

  const toggleThought = async () => {
    try {
      const d = await (await fetch(`${API_URL}/config/thought`, {method:"POST"})).json();
      setThoughtEnabled(d.internal_thought_enabled);
    } catch(_) {}
  };

  const toggleLongThinking = async () => {
    try {
      const d = await (await fetch(`${API_URL}/config/long_thinking`, {method:"POST"})).json();
      setLongThinking(d.long_thinking_enabled);
    } catch(_) {}
  };

  const toggleSeparateThought = async () => {
    try {
      setServerReady(false);
      const d = await (await fetch(`${API_URL}/config/separate_thought`, {method:"POST"})).json();
      setSeparateThought(d.separate_thought_model);
      if (window.require) {
        const { ipcRenderer } = window.require("electron");
        ipcRenderer.send("restart-backend");
      }
    } catch(_) { setServerReady(true); }
  };

  const toggleDevice = async () => {
    try {
      setServerReady(false);
      const d = await (await fetch(`${API_URL}/config/device`, {method:"POST"})).json();
      setDevice(d.device);
      if (window.require) {
        const { ipcRenderer } = window.require("electron");
        ipcRenderer.send("restart-backend");
      }
    } catch(_) { setServerReady(true); }
  };

  const triggerReflect = async () => {
    try { await fetch(`${API_URL}/reflect`, {method:"POST"}); fetchAll(); } catch(_) {}
  };

  const statusText = !serverReady ? "Memuat model…"
    : !connected   ? "Menghubungkan…"
    : thinking     ? (thought?.is_long_thinking ? "Berpikir dalam…" : "Berpikir…")
    : streaming    ? "Mengetik…"
    : "Online";

  const emoUser    = getEmo(userEmotion.user_emotion);
  const emoAsta    = getEmo(astaEmotion.current_emotion || astaEmotion.mood);
  const togglePanel = (name) => setPanel(p => p===name ? null : name);

  return (
    <div style={S.root}>
      {thought?.note && noteVisible && (
        <div className="note-bubble" style={S.noteBubbleFixed}>
          <div style={{fontSize:13,fontWeight:800,color:"var(--green)",marginBottom:4,letterSpacing:"0.05em"}}>
            {thought.is_long_thinking ? "✦ Deep Decision Directive" : "Decision Directive"}
            {thought.is_long_thinking && (
              <span style={{marginLeft:8,fontSize:10,color:"var(--purple)",fontWeight:500,fontFamily:"var(--mono)"}}>LONG THINK</span>
            )}
          </div>
          {thought.note}
          {thought.is_long_thinking && thought.hidden_need && (
            <div style={{marginTop:8,paddingTop:8,borderTop:"1px dashed var(--border)",fontSize:11,color:"var(--purple)",fontStyle:"italic"}}>
              ◈ Hidden need: {thought.hidden_need}
            </div>
          )}
        </div>
      )}

      <div style={S.topBar}>
        <TopBtn active={panel==="thought"}  onClick={()=>togglePanel("thought")}  icon="⟡"  label="Thought"  />
        <TopBtn active={panel==="self"}     onClick={()=>togglePanel("self")}     icon="◉"  label="Asta"     />
        <TopBtn active={panel==="memory"}   onClick={()=>togglePanel("memory")}   icon="◈"  label="Memory"   />
        {thought?.note && (
          <TopBtn
            active={noteVisible}
            onClick={()=>setNoteVisible(v=>!v)}
            onMouseEnter={()=>setNoteVisible(true)}
            onMouseLeave={()=>setNoteVisible(false)}
            icon="#" label="Action"
          />
        )}
        <TopBtn active={panel==="terminal"} onClick={()=>togglePanel("terminal")} icon=">_" label="Terminal" />
        <TopBtn active={panel==="stats"}    onClick={()=>togglePanel("stats")}    icon="◷"  label="Stats"    />
        <div style={{flex:1}}/>

        <div style={{display:"flex",alignItems:"center",gap:7,opacity:thoughtEnabled?1:0.35,transition:"opacity .2s"}}>
          <span style={{fontSize:11,color:longThinking?"var(--purple)":"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>
            {longThinking ? "✦ LONG THINK" : "✦ THINK"}
          </span>
          <button
            className={`dm-toggle${longThinking ? " is-on-purple" : ""}`}
            onClick={thoughtEnabled ? toggleLongThinking : undefined}
            title="Toggle Long Thinking"
            style={{WebkitAppRegion:"no-drag",cursor:thoughtEnabled?"pointer":"not-allowed"}}
          />
        </div>
        <Divider/>
        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{fontSize:11,color:thoughtEnabled?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>
            {thoughtEnabled ? "⟡ THOUGHT ON" : "⟡ OFF"}
          </span>
          <button
            className={`dm-toggle${thoughtEnabled ? " is-on" : ""}`}
            onClick={toggleThought}
            title="Toggle Internal Thought Logic"
            style={{WebkitAppRegion:"no-drag"}}
          />
        </div>
        <Divider/>
        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{fontSize:11,color:separateThought?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>
            {separateThought ? "❐ DUAL MODEL" : "❐ SHARED"}
          </span>
          <button
            className={`dm-toggle${separateThought ? " is-on" : ""}`}
            onClick={toggleSeparateThought}
            title="Toggle Separate 3B Thought Model (Reloads Model)"
            style={{WebkitAppRegion:"no-drag"}}
          />
        </div>
        <Divider/>
        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{fontSize:11,color:device==="gpu"?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>
            {device==="gpu" ? "CUDA" : "CPU"}
          </span>
          <button
            className={`dm-toggle${device==="gpu" ? " is-on" : ""}`}
            onClick={toggleDevice}
            title="Toggle CUDA Acceleration (Reloads Model)"
            style={{WebkitAppRegion:"no-drag"}}
          />
        </div>
        <Divider/>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:15,color:"var(--muted)",fontFamily:"var(--mono)",userSelect:"none"}}>{darkMode?"☾":"☀"}</span>
          <button
            className={`dm-toggle${darkMode ? " is-on" : ""}`}
            onClick={()=>setDarkMode(p=>!p)}
            title="Dark mode"
            style={{WebkitAppRegion:"no-drag"}}
          />
        </div>
      </div>

      <div style={S.layout}>
        <SidePanel visible={panel==="thought"} side="left"  title="Internal Thought" icon="⟡" width={285}>
          <ThoughtPanel thought={thought} thinking={thinking} modelInfo={modelInfo} />
        </SidePanel>
        <SidePanel visible={panel==="self"}    side="left"  title="Asta Self-Model"  icon="◉" width={270}>
          <SelfPanel selfModel={selfModel} astaEmotion={astaEmotion} onReflect={triggerReflect} />
        </SidePanel>

        <div style={S.chatCol}>
          <div style={S.header}>
            <div style={S.hLeft}>
              <Avatar emoAsta={emoAsta}/>
              <div>
                <div style={S.hName}>Asta</div>
                <div style={S.hSub}>{statusText}</div>
              </div>
            </div>
            <div style={S.hRight}>
              <AstaEmoBadge asta={astaEmotion} emo={emoAsta}/>
              <UserEmoBadge user={userEmotion} emo={emoUser}/>
              <button onClick={()=>fetch(`${API_URL}/save`,{method:"POST"}).then(fetchAll)} style={S.saveBtn} title="Simpan sesi">+</button>
            </div>
          </div>

          <div style={S.msgList}>
            {messages.length === 0 && (
              <div style={S.empty}>
                <div style={{fontSize:44,marginBottom:14,animation:"pulse 3s ease infinite"}}>{emoAsta.emoji}</div>
                <div style={{fontSize:18,fontWeight:500}}>Halo, Aditiya~</div>
                <div style={{fontSize:14,color:"var(--text)",marginTop:5}}>Asta siap ngobrol denganmu.</div>
              </div>
            )}
            {messages.map(m => (
              <Bubble key={m.id} msg={m} isStreaming={streaming && m.id===msgIdRef.current}/>
            ))}
            {thinking && <ThinkingBubble isLong={thought?.is_long_thinking}/>}
            <div ref={bottomRef}/>
          </div>

          <div style={S.inputWrap}>
            <div style={S.inputRow}>
              <textarea
                ref={mainInputRef}
                value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{ if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();} }}
                placeholder="Tulis pesan…"
                style={S.textarea} rows={1}
                disabled={!connected||thinking||streaming}
              />
              <button onClick={send} disabled={!connected||thinking||streaming||!input.trim()}
                style={{...S.sendBtn,opacity:(!connected||thinking||streaming||!input.trim())?0.6:1}}>↑</button>
            </div>
            <div style={S.hint}>Enter kirim · Shift+Enter baris baru</div>
          </div>
        </div>

        <SidePanel visible={panel==="memory"}   side="right" title="Memory"       icon="◈"  width={260}>
          <MemoryPanel memory={memory} onRefresh={fetchAll}/>
        </SidePanel>
        <SidePanel visible={panel==="terminal"} side="right" title="Terminal"     icon=">_" width={450} noPadding>
          <TerminalPanel visible={panel==="terminal"} onMessage={handleTerminalMessage}/>
        </SidePanel>
        <SidePanel visible={panel==="stats"}    side="right" title="System Stats" icon="◷"  width={200}>
          <div style={{display:"flex",flexDirection:"column",gap:10}}>
            <StatBar label="CPU"  value={sysStats.cpu}  color="var(--blue)"/>
            <StatBar label="RAM"  value={sysStats.ram}  color="var(--purple)"/>
            <StatBar label="DISK" value={sysStats.disk} color="var(--green)"/>
          </div>
        </SidePanel>
      </div>
    </div>
  );
}

// ── Primitives ────────────────────────────────────────────────────────────────

function Divider() {
  return <div style={{width:1,height:18,background:"var(--border)",flexShrink:0}}/>;
}

function TopBtn({ active, onClick, icon, label, onMouseEnter, onMouseLeave }) {
  const isDark = document.documentElement.classList.contains("dark");
  return (
    <button onClick={onClick} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}
      style={{
        display:"flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:99,
        background:active?"var(--asta)":"var(--surface)",
        color:active?(isDark?"#1a1816":"#f5f0eb"):"var(--muted)",
        border:`1px solid ${active?"var(--asta)":"var(--border)"}`,
        fontSize:12,fontFamily:"var(--font)",fontWeight:500,cursor:"pointer",
        transition:"all var(--ease)",WebkitAppRegion:"no-drag",
      }}>
      {icon} {label}
    </button>
  );
}

function SidePanel({ visible, side, title, icon, width=260, noPadding=false, children }) {
  return (
    <div style={{width:visible?width:0,minWidth:visible?width:0,overflow:"hidden",flexShrink:0,transition:"width .3s cubic-bezier(0.4,0,0.2,1),min-width .3s cubic-bezier(0.4,0,0.2,1)"}}>
      <div style={{width,height:"100%",background:"var(--surface)",borderLeft:side==="right"?"1px solid var(--border)":"none",borderRight:side==="left"?"1px solid var(--border)":"none",display:"flex",flexDirection:"column",opacity:visible?1:0,transition:"opacity .3s"}}>
        <div style={{padding:"14px 16px 10px",borderBottom:"1px solid var(--border)",display:"flex",alignItems:"center",gap:8,fontSize:11,fontWeight:600,letterSpacing:"0.06em",color:"var(--muted)",textTransform:"uppercase",flexShrink:0}}>
          {icon} {title}
        </div>
        <div style={{flex:1,overflowY:"auto",padding:noPadding?0:14}}>
          {children}
        </div>
      </div>
    </div>
  );
}

// ── Terminal ──────────────────────────────────────────────────────────────────

function TerminalPanel({ visible, onMessage }) {
  const [lines,  setLines]  = useState(["Asta Terminal Ready.", ""]);
  const [input,  setInput]  = useState("");
  const wsRef    = useRef(null);
  const scrollRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    if (visible) setTimeout(()=>inputRef.current?.focus(), 300);
  }, [visible]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8001");
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type==="clear")       setLines([]);
      else if (msg.type==="output") setLines(p=>[...p, msg.data]);
      else if (msg.type==="signal" && window.require) {
        const { ipcRenderer } = window.require("electron");
        if (msg.data==="start-backend") ipcRenderer.send("start-backend");
        if (msg.data==="stop-backend")  ipcRenderer.send("stop-backend");
      } else if (msg.type==="stats" && onMessage) onMessage(msg);
    };
    ws.onclose = ()=>setLines(p=>[...p, "[Disconnected]"]);
    if (window.require) {
      const { ipcRenderer } = window.require("electron");
      const h = (_,d) => setLines(p=>[...p, d]);
      ipcRenderer.on("backend-out", h);
      ipcRenderer.on("backend-err", h);
      return ()=>{
        ws.close();
        ipcRenderer.removeListener("backend-out", h);
        ipcRenderer.removeListener("backend-err", h);
      };
    }
    return ()=>ws.close();
  }, [onMessage]);

  useEffect(()=>{ scrollRef.current?.scrollIntoView({behavior:"smooth"}); }, [lines]);

  const runCmd = () => {
    if (!input.trim() || !wsRef.current) return;
    setLines(p=>[...p, `> ${input}`]);
    wsRef.current.send(input);
    setInput("");
  };

  return (
    <div style={{background:"#0c0c0c",color:"#fff",fontFamily:"var(--mono)",fontSize:11,height:"100%",display:"flex",flexDirection:"column",padding:"12px 16px",lineHeight:1.4,textAlign:"left"}}>
      <div className="hide-scrollbar" style={{flex:1,overflowY:"auto",overflowX:"auto",whiteSpace:"pre",marginBottom:10}}>
        {lines.map((l,i)=><div key={i} style={{marginBottom:1}}>{l}</div>)}
        <div ref={scrollRef}/>
      </div>
      <div style={{display:"flex",gap:5,borderTop:"1px solid #333",paddingTop:10}}>
        <span>$</span>
        <input ref={inputRef} value={input} onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>e.key==="Enter"&&runCmd()}
          style={{background:"transparent",border:"none",color:"#fff",fontFamily:"var(--mono)",fontSize:11,outline:"none",flex:1}}/>
      </div>
    </div>
  );
}

// ── Chat components ───────────────────────────────────────────────────────────

function Avatar({ emoAsta }) {
  const isSym = emoAsta.emoji==="*";
  return (
    <div style={{position:"relative",marginRight:12}}>
      <div style={{width:46,height:46,borderRadius:12,background:`linear-gradient(135deg,${emoAsta.color}22,${emoAsta.color}08)`,border:`1px solid ${emoAsta.color}25`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:isSym?32:24,transition:"all .4s",overflow:"hidden"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"center",transform:isSym?"translateY(6px)":"none",color:emoAsta.color,lineHeight:0}}>
          {emoAsta.emoji}
        </div>
      </div>
    </div>
  );
}

function AstaEmoBadge({ asta, emo }) {
  const pct = Math.round(((asta.mood_score||0)+1)/2*100);
  return (
    <div title={`Mood: ${asta.mood} | Affection: ${(asta.affection_level||0.7).toFixed(2)}`}
      style={{display:"flex",alignItems:"center",gap:6,padding:"4px 10px",textTransform:"capitalize",borderRadius:99,background:`${emo.color}14`,border:`1px solid ${emo.color}30`,fontSize:11,fontFamily:"var(--mono)",color:emo.color,transition:"all .4s"}}>
      <span style={{fontSize:14}}>{emo.emoji}</span>
      <span>{emo.label}</span>
      <div style={{width:32,height:5,background:"var(--border)",borderRadius:99,overflow:"hidden"}}>
        <div className="bar-fill" style={{height:"100%",width:`${pct}%`,background:emo.color,borderRadius:99}}/>
      </div>
    </div>
  );
}

function UserEmoBadge({ user, emo }) {
  return (
    <div title="Emosi user"
      style={{display:"flex",alignItems:"center",gap:5,padding:"4px 10px",textTransform:"capitalize",borderRadius:99,background:"var(--surface2)",border:"1px solid var(--border)",fontSize:11,fontFamily:"var(--mono)",color:"var(--muted)"}}>
      <span style={{color:emo.color}}>{emo.emoji}</span>
      <span>{user.user_emotion}</span>
    </div>
  );
}

function Bubble({ msg, isStreaming }) {
  const isUser = msg.role==="user";
  return (
    <div style={{display:"flex",justifyContent:isUser?"flex-end":"flex-start",padding:"3px 0",animation:`${isUser?"slideR":"slideL"} .28s ease`}}>
      <div style={{display:"flex",flexDirection:"column",alignItems:isUser?"flex-end":"flex-start",maxWidth:"68%",gap:6}}>
        {!isUser && msg.webSearch && <WebSearchSubBubble webSearch={msg.webSearch}/>}
        {!isUser && msg.isLong && (
          <div style={{alignSelf:"flex-start",padding:"2px 9px",borderRadius:99,background:"var(--purple)18",border:"1px solid var(--purple)44",fontSize:10,fontFamily:"var(--mono)",color:"var(--purple)",letterSpacing:"0.05em"}}>
            ✦ deep think
          </div>
        )}
        <div style={{
          padding:"11px 16px",
          borderRadius:isUser?"16px 4px 16px 16px":"4px 16px 16px 16px",
          background:isUser?"var(--asta)":"var(--surface)",
          color:isUser?"#f5f0eb":"var(--text)",
          fontSize:13,
          boxShadow:"var(--shadow)",
          border:isUser?"none":"1px solid var(--border)",
          wordBreak:"break-word",
          textAlign:"left",
          width:"100%",
        }}>
          {/* FIX #4: Gunakan MessageContent untuk rendering markdown */}
          {isStreaming && msg.tokens
            ? <>{msg.tokens.map(t=><span key={t.id} className="stream-token">{t.text}</span>)}<Cursor/></>
            : isUser
              ? <div className="msg-body"><span className="msg-line">{msg.content}</span></div>
              : <><MessageContent content={msg.content}/>{isStreaming && <Cursor/>}</>
          }
        </div>
      </div>
    </div>
  );
}

function WebSearchSubBubble({ webSearch }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <div style={{width:"94%",border:"1px solid var(--border)",borderRadius:"12px 12px 12px 4px",background:"var(--surface)",boxShadow:"var(--shadow)",overflow:"hidden",animation:"pullOut .25s ease",position:"relative"}}>
      <div style={{position:"absolute",left:10,bottom:-8,width:14,height:14,background:"var(--surface)",borderRight:"1px solid var(--border)",borderBottom:"1px solid var(--border)",transform:"rotate(45deg)"}}/>
      <button onClick={()=>setCollapsed(p=>!p)}
        style={{width:"100%",display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 11px",border:"none",background:"transparent",cursor:"pointer",fontSize:11,fontFamily:"var(--mono)",color:"var(--muted)",position:"relative",zIndex:1}}>
        <span style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{color:"var(--accent)",fontSize:18,marginBottom:2}}>⌕</span>
          <span style={{color:"var(--text)",fontWeight:500}}>Web search: {webSearch.query}</span>
        </span>
        <span style={{color:"var(--accent)",fontSize:25,marginBottom:2}}>{collapsed?"▾":"▴"}</span>
      </button>
      <div style={{maxHeight:collapsed?0:160,overflow:"hidden",transition:"max-height var(--ease)",position:"relative",zIndex:1}}>
        <div style={{maxHeight:140,overflowY:"auto",padding:"10px 13px",fontSize:12,textAlign:"left",lineHeight:1.55,color:"var(--text)",whiteSpace:"pre-wrap",wordBreak:"break-word",borderTop:"1px dashed var(--border)"}}>
          {webSearch.result}
        </div>
      </div>
    </div>
  );
}

const Cursor = () => (
  <span style={{display:"inline-block",width:7,height:14,background:"var(--accent)",borderRadius:2,marginLeft:2,verticalAlign:"text-bottom",animation:"blink .8s step-end infinite"}}/>
);

function ThinkingBubble({ isLong }) {
  return (
    <div style={{display:"flex",padding:"3px 0",animation:"fadeIn .3s ease"}}>
      <div style={{padding:"13px 18px",borderRadius:"4px 16px 16px 16px",background:"var(--surface)",border:`1px solid ${isLong?"var(--purple)44":"var(--border)"}`,boxShadow:"var(--shadow)",display:"flex",gap:6,alignItems:"center"}}>
        {isLong && <span style={{fontSize:11,color:"var(--purple)",fontFamily:"var(--mono)",marginRight:4}}>✦</span>}
        {[0,.18,.36].map(d=>(
          <div key={d} style={{width:6,height:6,borderRadius:"50%",background:isLong?"var(--purple)":"var(--accent)",animation:`pulse 1.2s ease-in-out ${d}s infinite`}}/>
        ))}
        {isLong && <span style={{fontSize:10,color:"var(--purple)",fontFamily:"var(--mono)",marginLeft:4}}>deep thinking…</span>}
      </div>
    </div>
  );
}

function StatBar({ label, value, color }) {
  return (
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:"var(--mono)",color:"var(--muted)",marginBottom:3}}>
        <span>{label}</span><span style={{color}}>{value}%</span>
      </div>
      <div style={{height:6,background:"var(--surface2)",border:"1px solid var(--border)",borderRadius:99,overflow:"hidden"}}>
        <div className="bar-fill" style={{height:"100%",width:`${value}%`,background:color,borderRadius:99}}/>
      </div>
    </div>
  );
}

// ── Thought Panel ─────────────────────────────────────────────────────────────

function ThoughtPanel({ thought, thinking, modelInfo }) {
  if (thinking) return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",paddingTop:40}}>
      <div style={{width:18,height:18,borderRadius:"50%",border:`2px solid var(--border)`,borderTop:`2px solid ${thought?.is_long_thinking?"var(--purple)":"var(--accent)"}`,animation:"spin .8s linear infinite",marginBottom:10}}/>
      <span style={{fontSize:11,color:"var(--muted)"}}>{modelInfo.dual_model?`Berpikir… (${modelInfo.thought_model})`:"Berpikir…"}</span>
    </div>
  );
  if (!thought) return <div style={{color:"var(--muted)",fontSize:12,textAlign:"center",paddingTop:40}}>Belum ada thought</div>;

  const isLong = thought.is_long_thinking;
  const mi     = thought.model_info || modelInfo;
  const stepLabel = isLong ? "F" : "S";

  const steps = [
    {
      num:"1", label:"PERCEPTION",
      rows:[
        {k:"Topic",    v:thought.topic},
        {k:"Sentiment",v:thought.sentiment},
        {k:"Urgency",  v:thought.urgency},
        isLong&&thought.complexity&&{k:"Complexity",v:thought.complexity,color:"var(--purple)"},
        isLong&&thought.hidden_need&&{k:"Hidden Need",v:thought.hidden_need,color:"var(--purple)",italic:true},
      ].filter(Boolean),
    },
    {
      num:"2", label:"SELF-CHECK", color:"var(--rose)",
      rows:[
        {k:"Emosi Asta", v:thought.asta_emotion, color:getEmo(thought.asta_emotion).color},
        {k:"Trigger",    v:thought.asta_trigger},
        {k:"Ekspresikan",v:thought.should_express?"Ya":"Tidak"},
      ],
    },
    {
      num:"3", label:"MEMORY", color:"var(--blue)",
      rows:[
        {k:"Search",    v:thought.need_search?"✓ Ya":"✗ Tidak", mono:true},
        thought.search_query&&{k:"Query",     v:thought.search_query, mono:true},
        {k:"Recall",    v:thought.recall_topic||"–"},
        {k:"Reasoning", v:thought.reasoning||"–"},
      ].filter(Boolean),
    },
    {
      num:"4", label:"DECISION", color:"var(--green)",
      rows:[
        {k:"Tone",  v:thought.tone},
        {k:"Style", v:thought.response_style||"normal"},
        isLong&&thought.response_structure&&{k:"Structure",v:thought.response_structure,color:"var(--purple)"},
        isLong&&thought.anticipated_followup&&{k:"Followup",v:thought.anticipated_followup,color:"var(--muted)",italic:true},
      ].filter(Boolean),
    },
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:8,animation:"fadeIn .3s ease"}}>
      {isLong && (
        <div className="long-think-badge" style={{padding:"6px 10px",borderRadius:"var(--rs)",background:"var(--purple)12",border:"1px solid var(--purple)44",fontSize:11,fontFamily:"var(--mono)",color:"var(--purple)",display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:14}}>✦</span>
          <span>Long Thinking — Analisis Mendalam (2-pass)</span>
        </div>
      )}
      {!isLong && (
        <div style={{padding:"4px 9px",borderRadius:"var(--rs)",background:"var(--surface2)",border:"1px solid var(--border)",fontSize:10,fontFamily:"var(--mono)",color:"var(--muted)"}}>
          2-pass thought
        </div>
      )}
      {mi.dual_model && (
        <div style={{padding:"6px 9px",borderRadius:"var(--rs)",border:"1px solid var(--border)"}}>
          <div style={S.cardLabel}>Pipeline</div>
          <div style={{display:"flex",alignItems:"center",gap:6,marginTop:3,flexWrap:"wrap"}}>
            <span style={{padding:"3px 7px",borderRadius:99,fontSize:10,fontFamily:"var(--mono)",fontWeight:600,background:"#7a9ec722",color:"#7a9ec7",border:"1px solid #7a9ec733"}}>⟡ {mi.thought_model}</span>
            <span style={{fontSize:10,color:"var(--muted)"}}>→</span>
            <span style={{padding:"3px 7px",borderRadius:99,fontSize:10,fontFamily:"var(--mono)",fontWeight:600,background:"var(--accent)22",color:"var(--accent)",border:"1px solid var(--accent)33"}}>↑ {mi.response_model}</span>
          </div>
        </div>
      )}
      {steps.map(step=>(
        <div key={step.num} style={{borderRadius:"var(--rs)",border:`1px solid ${step.color||"var(--border)"}33`,overflow:"hidden"}}>
          <div style={{padding:"5px 10px",background:`${step.color||"var(--accent)"}12`,borderBottom:`1px solid ${step.color||"var(--border)"}18`,fontSize:10.5,fontWeight:700,letterSpacing:"0.08em",color:step.color||"var(--muted)",fontFamily:"var(--mono)",textAlign:"left"}}>
            {stepLabel}{step.num} · {step.label}
          </div>
          <div style={{padding:"0 10px",display:"flex",flexDirection:"column"}}>
            {step.rows.map((row,i)=>row&&(
              <div key={i} style={{display:"grid",gridTemplateColumns:"90px 1fr",gap:10,fontSize:11,alignItems:"start",textAlign:"left"}}>
                <span style={{color:"var(--muted)",fontWeight:500}}>{row.k}</span>
                <span style={{fontFamily:row.mono?"var(--mono)":"var(--font)",color:row.color||"var(--text)",textAlign:"left",wordBreak:"break-word",lineHeight:2.5,fontStyle:row.italic?"italic":"normal"}}>
                  {row.v||"–"}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Self Panel ────────────────────────────────────────────────────────────────

function SelfPanel({ selfModel, astaEmotion, onReflect }) {
  if (!selfModel) return <div style={{color:"var(--muted)",fontSize:12,textAlign:"center",paddingTop:40}}>Memuat…</div>;
  const emo    = getEmo(astaEmotion.current_emotion || astaEmotion.mood);
  const moodPct = Math.round(((astaEmotion.mood_score||0)+1)/2*100);
  const affPct  = Math.round((astaEmotion.affection_level||0.7)*100);
  const engPct  = Math.round((astaEmotion.energy_level||0.8)*100);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14,animation:"fadeIn .3s ease"}}>
      <div>
        <div style={S.sectionTitle}>Kondisi Emosional</div>
        <div style={{marginTop:7,padding:"10px 12px",borderRadius:"var(--rs)",background:`${emo.color}0e`,border:`1px solid ${emo.color}25`}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
            <span style={{fontSize:24}}>{emo.emoji}</span>
            <div>
              <div style={{fontSize:13,fontWeight:600,marginLeft:5,textAlign:"left",color:emo.color}}>{emo.label}</div>
              <div style={{fontSize:11,marginLeft:5,textAlign:"left",color:"var(--muted)",fontFamily:"var(--mono)"}}>{astaEmotion.mood}</div>
            </div>
          </div>
          {[
            {label:"Mood",      pct:moodPct, color:emo.color,    val:`${astaEmotion.mood_score>=0?"+":""}${(astaEmotion.mood_score||0).toFixed(2)}`},
            {label:"Affection", pct:affPct,  color:"var(--rose)", val:`${affPct}%`},
            {label:"Energy",    pct:engPct,  color:"var(--green)",val:`${engPct}%`},
          ].map(b=>(
            <div key={b.label} style={{marginBottom:6}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"var(--muted)",marginBottom:2}}>
                <span>{b.label}</span><span style={{fontFamily:"var(--mono)",color:b.color}}>{b.val}</span>
              </div>
              <div style={{height:4,background:"var(--border)",borderRadius:99,overflow:"hidden"}}>
                <div className="bar-fill" style={{height:"100%",width:`${b.pct}%`,background:b.color,borderRadius:99}}/>
              </div>
            </div>
          ))}
        </div>
      </div>
      {selfModel.identity?.nilai_inti?.length>0&&(
        <div>
          <div style={S.sectionTitle}>Nilai Inti</div>
          <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:6}}>
            {selfModel.identity.nilai_inti.map((v,i)=>(
              <div key={i} style={{fontSize:11,padding:"4px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)",color:"var(--text)"}}>◦ {v}</div>
            ))}
          </div>
        </div>
      )}
      {selfModel.memories_of_self?.length>0&&(
        <div>
          <div style={S.sectionTitle}>Kenangan Diri</div>
          <div style={{display:"flex",flexDirection:"column",gap:4,marginTop:6}}>
            {selfModel.memories_of_self.map((m,i)=>(
              <div key={i} style={{fontSize:11,padding:"5px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <div style={{color:"var(--muted)",fontFamily:"var(--mono)",fontSize:9,marginBottom:2}}>{m.timestamp?.slice(0,10)}</div>
                <div>{m.content}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {selfModel.last_reflection&&(
        <div>
          <div style={S.sectionTitle}>Refleksi Terakhir</div>
          <div style={{marginTop:6,padding:"8px 10px",borderRadius:"var(--rs)",background:"var(--surface2)",border:"1px solid var(--border)",fontSize:12,lineHeight:1.6}}>
            {selfModel.last_reflection.summary}
          </div>
          {selfModel.last_reflection.growth_note&&(
            <div style={{marginTop:4,fontSize:11,color:"var(--accent)",fontStyle:"italic",padding:"0 2px"}}>✦ {selfModel.last_reflection.growth_note}</div>
          )}
          <div style={{fontSize:10,color:"var(--muted)",marginTop:4,fontFamily:"var(--mono)"}}>{selfModel.reflection_count} refleksi tersimpan</div>
        </div>
      )}
      {selfModel.growth_log?.length>0&&(
        <div>
          <div style={S.sectionTitle}>Growth Log</div>
          <div style={{display:"flex",flexDirection:"column",gap:3,marginTop:6}}>
            {selfModel.growth_log.slice(-3).map((g,i)=>(
              <div key={i} style={{fontSize:11,padding:"4px 8px",borderRadius:6,background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <span style={{color:"var(--muted)",fontFamily:"var(--mono)",fontSize:9}}>{g.timestamp?.slice(0,10)} </span>{g.entry}
              </div>
            ))}
          </div>
        </div>
      )}
      <button onClick={onReflect} style={{...S.refreshBtn,borderColor:"var(--accent)44",color:"var(--accent)"}}>✦ Refleksi Sekarang</button>
    </div>
  );
}

// ── Memory Panel ──────────────────────────────────────────────────────────────

function MemoryPanel({ memory, onRefresh }) {
  if (!memory) return <div style={{color:"var(--muted)",fontSize:13,textAlign:"center",paddingTop:40}}>Memuat…</div>;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:16,animation:"fadeIn .3s ease"}}>
      {memory.profile?.preferensi?.length>0&&(
        <div>
          <div style={S.sectionTitle}>Preferensi</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:5,marginTop:7}}>
            {memory.profile.preferensi.map((p,i)=><span key={i} style={S.tag}>{p}</span>)}
          </div>
        </div>
      )}
      {memory.recent_facts&&(
        <div>
          <div style={S.sectionTitle}>Fakta Terbaru</div>
          <div style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--muted)",lineHeight:1.8,marginTop:7,whiteSpace:"pre-wrap"}}>{memory.recent_facts}</div>
        </div>
      )}
      {memory.core&&(
        <div>
          <div style={S.sectionTitle}>Core Summary</div>
          <div style={{fontSize:12,lineHeight:1.7,marginTop:7,padding:"10px 12px",background:"var(--surface2)",borderRadius:"var(--rs)"}}>{memory.core}</div>
        </div>
      )}
      {memory.sessions?.length>0&&(
        <div>
          <div style={S.sectionTitle}>Sesi Tersimpan</div>
          <div style={{display:"flex",flexDirection:"column",gap:6,marginTop:7}}>
            {memory.sessions.map((s,i)=>(
              <div key={i} style={{padding:"6px 9px",borderRadius:"var(--rs)",background:"var(--surface2)",border:"1px solid var(--border)"}}>
                <div style={{fontSize:10,color:"var(--muted)",fontFamily:"var(--mono)"}}>
                  {new Date(s.timestamp).toLocaleString("id-ID",{dateStyle:"short",timeStyle:"short"})} · {s.facts} fakta
                </div>
                <div style={{fontSize:11,marginTop:3,lineHeight:1.5}}>{s.preview}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <button onClick={onRefresh} style={S.refreshBtn}>↻ Refresh</button>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const S = {
  root:           {display:"flex",flexDirection:"column",width:"100%",height:"100vh",overflow:"hidden",background:"var(--bg)"},
  topBar:         {display:"flex",alignItems:"center",gap:8,padding:"10px 20px",paddingTop:"35px",flexShrink:0,borderBottom:"1px solid var(--border)",background:"var(--bg)",WebkitAppRegion:"drag"},
  layout:         {flex:1,display:"flex",overflow:"hidden",minHeight:0},
  chatCol:        {flex:1,display:"flex",flexDirection:"column",minWidth:0},
  header:         {display:"flex",alignItems:"center",justifyContent:"space-between",padding:"14px 24px",flexShrink:0,borderBottom:"1px solid var(--border)",background:"var(--surface)"},
  hLeft:          {display:"flex",alignItems:"center"},
  hName:          {fontSize:17,textAlign:"left",fontWeight:600,letterSpacing:"-0.01em",lineHeight:1.4},
  hSub:           {fontSize:12,textAlign:"left",color:"var(--muted)",marginTop:1,fontFamily:"var(--mono)",lineHeight:1.3},
  hRight:         {display:"flex",alignItems:"center",gap:10},
  msgList:        {flex:1,overflowY:"auto",padding:"24px 32px",display:"flex",flexDirection:"column",gap:4,minHeight:0},
  noteBubbleFixed:{position:"absolute",left:395,top:32,width:320,padding:"14px 18px",background:"var(--surface)",border:"1.5px solid var(--green)",borderRadius:"16px",boxShadow:"0 10px 40px rgba(0,0,0,0.15)",zIndex:1000,fontSize:12,lineHeight:1.55,color:"var(--text)",fontStyle:"italic",textAlign:"left"},
  empty:          {display:"flex",flex:1,flexDirection:"column",alignItems:"center",justifyContent:"center",opacity:0.55,animation:"fadeIn .6s ease"},
  inputWrap:      {padding:"14px 24px 18px",flexShrink:0,background:"var(--bg)"},
  inputRow:       {display:"flex",gap:12,alignItems:"flex-end"},
  textarea:       {flex:1,resize:"none",padding:"12px 16px",borderRadius:"var(--r)",border:"1.5px solid var(--surface)",background:"var(--surface)",fontSize:15,fontFamily:"var(--font)",color:"var(--text)",lineHeight:1.6,outline:"none",maxHeight:140,overflowY:"auto"},
  sendBtn:        {width:50,height:50,borderRadius:"50%",background:"var(--asta)",color:"#f5f0eb",border:"none",fontSize:18,cursor:"pointer",transition:"all var(--ease)",flexShrink:0,display:"flex",alignItems:"center",justifyContent:"center",fontWeight:600},
  hint:           {fontSize:11,color:"var(--muted)",marginTop:7,fontFamily:"var(--mono)"},
  saveBtn:        {width:36,height:36,borderRadius:"50%",background:"transparent",border:"1px solid var(--border)",color:"var(--muted)",fontSize:14,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"},
  modelBadge:     {display:"flex",alignItems:"center",gap:6,padding:"4px 10px",borderRadius:99,background:"var(--surface2)",border:"1px solid var(--border)",fontSize:11,fontFamily:"var(--mono)",fontWeight:500},
  cardLabel:      {fontSize:11,color:"var(--muted)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:2,fontWeight:600},
  sectionTitle:   {fontSize:11,fontWeight:600,color:"var(--muted)",textTransform:"uppercase",marginBottom:3,letterSpacing:"0.06em"},
  tag:            {padding:"4px 11px",borderRadius:99,background:"var(--tag-bg)",fontSize:12,color:"var(--text)",fontWeight:500},
  refreshBtn:     {width:"100%",padding:"9px",borderRadius:"var(--rs)",border:"1px solid var(--border)",background:"transparent",cursor:"pointer",fontSize:13,color:"var(--muted)",fontFamily:"var(--font)"},
};