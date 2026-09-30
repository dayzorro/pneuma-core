"""Embedded single-page web UI for the chat service.

Kept as a Python string (not a static file) so it ships with the wheel
without extra packaging configuration.
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Pneuma Core · 与角色对话</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #171a23;
    --panel-2: #1e2230;
    --line: #2a2f3f;
    --text: #e7e9ee;
    --muted: #8b93a7;
    --accent: #6c8cff;
    --accent-2: #8b6cff;
    --user: #2b3350;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: var(--bg); color: var(--text);
    font: 15px/1.6 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 20px; border-bottom: 1px solid var(--line);
    background: var(--panel);
  }
  header .avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; color: #fff;
  }
  header .meta { flex: 1; min-width: 0; }
  header .name { font-weight: 600; }
  header .sub { font-size: 12px; color: var(--muted); }
  .emotion {
    font-size: 12px; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); white-space: nowrap;
  }
  button {
    font: inherit; cursor: pointer; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel-2); color: var(--text);
    padding: 8px 14px; transition: .15s;
  }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    border: none; color: #fff; font-weight: 600;
  }
  #setup {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 16px; padding: 24px;
  }
  #setup h1 { font-size: 22px; margin: 0; }
  #setup p { color: var(--muted); margin: 0; text-align: center; }
  #setup input {
    width: min(360px, 90vw); padding: 12px 16px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--panel-2);
    color: var(--text); font: inherit; outline: none;
  }
  #setup input:focus { border-color: var(--accent); }
  #chat { flex: 1; display: none; flex-direction: column; min-height: 0; }
  #log { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 78%; display: flex; flex-direction: column; gap: 4px; }
  .msg .who { font-size: 12px; color: var(--muted); }
  .bubble {
    padding: 10px 14px; border-radius: 14px; background: var(--panel-2);
    border: 1px solid var(--line); white-space: pre-wrap; word-break: break-word;
  }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.user .bubble { background: var(--user); border-color: #3a4570; }
  .extra { font-size: 12px; color: var(--muted); }
  .extra summary { cursor: pointer; outline: none; }
  .extra div { margin-top: 4px; padding-left: 10px; border-left: 2px solid var(--line); }
  .sys { align-self: center; font-size: 12px; color: #d98b8b; }
  footer {
    display: flex; gap: 10px; padding: 14px 20px;
    border-top: 1px solid var(--line); background: var(--panel);
  }
  footer textarea {
    flex: 1; resize: none; height: 46px; padding: 12px 14px;
    border-radius: 10px; border: 1px solid var(--line);
    background: var(--panel-2); color: var(--text); font: inherit; outline: none;
  }
  footer textarea:focus { border-color: var(--accent); }
  .typing { color: var(--muted); font-size: 13px; padding: 0 20px 8px; }
</style>
</head>
<body>
<header>
  <div class="avatar" id="avatar">?</div>
  <div class="meta">
    <div class="name" id="charName">加载中…</div>
    <div class="sub" id="charSub"></div>
  </div>
  <div class="emotion" id="emotion">情绪 —</div>
  <button id="endBtn" style="display:none">结束会话</button>
</header>

<section id="setup">
  <h1>先告诉我，你是谁？</h1>
  <p>设定你的名字后就可以开始对话。对话结束后，<br/>她才会把这段经历整理成记忆。</p>
  <input id="userName" placeholder="你的名字，例如：阿泽" autocomplete="off" />
  <button class="primary" id="startBtn">开始对话</button>
</section>

<section id="chat">
  <div id="log"></div>
  <div class="typing" id="typing" style="display:none">正在思考…</div>
  <footer>
    <textarea id="input" placeholder="说点什么…（Enter 发送，Shift+Enter 换行）"></textarea>
    <button class="primary" id="sendBtn">发送</button>
  </footer>
</section>

<script>
const $ = (id) => document.getElementById(id);
let started = false;

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

async function loadCharacter() {
  try {
    const c = await api("/api/character");
    $("charName").textContent = c.name;
    $("avatar").textContent = c.name.slice(0, 1);
    $("charSub").textContent = (c.profile || "").trim().split("\\n")[0];
  } catch (e) {
    $("charName").textContent = "无法连接服务";
  }
}

function addMessage(role, text, extra) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "user" : "char");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "你" : $("charName").textContent;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(who, bubble);
  if (extra) {
    const d = document.createElement("details");
    d.className = "extra";
    d.innerHTML = "<summary>查看内心</summary>";
    const inner = document.createElement("div");
    inner.textContent = extra;
    d.appendChild(inner);
    wrap.appendChild(d);
  }
  $("log").appendChild(wrap);
  $("log").scrollTop = $("log").scrollHeight;
}

function addSystem(text) {
  const d = document.createElement("div");
  d.className = "sys";
  d.textContent = "⚠ " + text;
  $("log").appendChild(d);
  $("log").scrollTop = $("log").scrollHeight;
}

function startStreamingMessage() {
  const wrap = document.createElement("div");
  wrap.className = "msg char";
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = $("charName").textContent;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.append(who, bubble);
  $("log").appendChild(wrap);
  $("log").scrollTop = $("log").scrollHeight;
  return {
    append(t) {
      bubble.textContent += t;
      $("log").scrollTop = $("log").scrollHeight;
    },
    attachExtra(extra) {
      if (!extra) return;
      const d = document.createElement("details");
      d.className = "extra";
      d.innerHTML = "<summary>查看内心</summary>";
      const inner = document.createElement("div");
      inner.textContent = extra;
      d.appendChild(inner);
      wrap.appendChild(d);
    },
    remove() {
      wrap.remove();
    },
  };
}

async function start() {
  const name = $("userName").value.trim();
  if (!name) return $("userName").focus();
  $("startBtn").disabled = true;
  try {
    const data = await api("/api/session/start", { user_id: "web-user", user_name: name });
    started = true;
    $("setup").style.display = "none";
    $("chat").style.display = "flex";
    $("endBtn").style.display = "block";
    addMessage("char", `你好，${name}。我是${data.character.name}。`, null);
    $("input").focus();
  } catch (e) {
    alert("启动失败：" + e.message);
    $("startBtn").disabled = false;
  }
}

async function send() {
  const text = $("input").value.trim();
  if (!text || !started) return;
  $("input").value = "";
  addMessage("user", text);
  $("sendBtn").disabled = true;
  $("typing").style.display = "block";
  const stream = startStreamingMessage();
  let gotDelta = false;
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let done = null;
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += decoder.decode(chunk.value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\\n\\n")) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const evt = JSON.parse(raw.slice(5).trim());
        if (evt.type === "delta") {
          $("typing").style.display = "none";
          gotDelta = true;
          stream.append(evt.text);
        } else if (evt.type === "done") {
          done = evt;
        } else if (evt.type === "error") {
          throw new Error(evt.message || "stream error");
        }
      }
    }
    $("typing").style.display = "none";
    if (!done) throw new Error("连接中断");
    const parts = [];
    if (done.thought) parts.push("内心： " + done.thought);
    if (done.action) parts.push("动作： " + done.action);
    if (!gotDelta) stream.append(done.reply || "");
    stream.attachExtra(parts.join("\\n") || null);
    (done.system_messages || []).forEach((m) => addSystem(m.message));
    if (done.emotion && done.emotion.label) {
      $("emotion").textContent = "情绪 " + done.emotion.label;
    }
  } catch (e) {
    $("typing").style.display = "none";
    if (!gotDelta) stream.remove();
    addSystem("请求失败：" + e.message);
  } finally {
    $("sendBtn").disabled = false;
    $("input").focus();
  }
}

async function end() {
  if (!started) return;
  $("endBtn").disabled = true;
  $("endBtn").textContent = "整理记忆中…";
  try {
    const r = await api("/api/session/end");
    addSystem(
      `会话结束。情节记忆 +${r.episodic_memories_saved}，语义记忆 +${r.semantic_updates_applied}，关系变化 ${r.relationship_changes}。`
    );
  } catch (e) {
    addSystem("结束失败：" + e.message);
  }
  started = false;
  $("endBtn").style.display = "none";
  $("sendBtn").disabled = true;
  $("input").disabled = true;
}

$("startBtn").onclick = start;
$("sendBtn").onclick = send;
$("endBtn").onclick = end;
$("userName").addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
loadCharacter();
</script>
</body>
</html>
"""
