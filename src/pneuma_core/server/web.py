"""Embedded single-page web UI for the chat service.

Kept as a Python string (not a static file) so it ships with the wheel
without extra packaging configuration.

界面按「门店服务前台」设计：品牌头部、值班信息、常见问题快捷入口、
转人工入口，以及可展开的「内心」面板（用于演示情感与记忆机制）。
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>华润万家 · 顾客服务前台</title>
<style>
  :root {
    --bg: #f4f5f7;
    --panel: #ffffff;
    --line: #e6e8ec;
    --text: #1f2329;
    --muted: #8a9099;
    --brand: #d7261e;
    --brand-dark: #b01c15;
    --brand-soft: #fff2f0;
    --user: #eef2fb;
    --user-line: #d8e1f5;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: var(--bg); color: var(--text);
    font: 15px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }

  /* ── 头部 ─────────────────────────────── */
  header {
    background: var(--panel); border-bottom: 1px solid var(--line);
  }
  .brandbar {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 20px; background: var(--brand); color: #fff;
    font-size: 13px; letter-spacing: .5px;
  }
  .brandbar .mark {
    width: 18px; height: 18px; border-radius: 50%;
    background: #ffd24a; display: flex; align-items: center; justify-content: center;
    color: var(--brand-dark); font-weight: 700; font-size: 11px;
  }
  .deskbar {
    display: flex; align-items: center; gap: 12px; padding: 12px 20px;
  }
  .avatar {
    width: 42px; height: 42px; border-radius: 50%; flex: none;
    background: linear-gradient(135deg, #ff8a5b, var(--brand));
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; color: #fff; font-size: 17px;
  }
  .meta { flex: 1; min-width: 0; }
  .name { font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .name .online {
    font-size: 12px; font-weight: 400; color: #1a9c5b;
    display: inline-flex; align-items: center; gap: 4px;
  }
  .name .online::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%; background: #1a9c5b;
  }
  .sub { font-size: 12px; color: var(--muted); }
  .chip {
    font-size: 12px; padding: 3px 10px; border-radius: 999px;
    border: 1px solid var(--line); color: var(--muted); white-space: nowrap;
  }
  .chip b { color: var(--brand); font-weight: 600; }

  button {
    font: inherit; cursor: pointer; border-radius: 8px;
    border: 1px solid var(--line); background: #fff; color: var(--text);
    padding: 8px 14px; transition: .15s;
  }
  button:hover:not(:disabled) { border-color: var(--brand); color: var(--brand); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.primary {
    background: var(--brand); border: 1px solid var(--brand);
    color: #fff; font-weight: 600;
  }
  button.primary:hover:not(:disabled) { background: var(--brand-dark); color: #fff; }

  /* ── 开场 ─────────────────────────────── */
  #setup {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 14px; padding: 24px;
  }
  #setup .welcome { font-size: 22px; font-weight: 600; }
  #setup .welcome span { color: var(--brand); }
  #setup p { color: var(--muted); margin: 0; text-align: center; max-width: 420px; }
  #setup input {
    width: min(360px, 90vw); padding: 12px 16px; border-radius: 10px;
    border: 1px solid var(--line); background: #fff;
    color: var(--text); font: inherit; outline: none;
  }
  #setup input:focus { border-color: var(--brand); }

  /* ── 对话 ─────────────────────────────── */
  #chat { flex: 1; display: none; flex-direction: column; min-height: 0; }
  #log { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 78%; display: flex; flex-direction: column; gap: 4px; }
  .msg .who { font-size: 12px; color: var(--muted); }
  .bubble {
    padding: 10px 14px; border-radius: 12px; background: var(--panel);
    border: 1px solid var(--line); white-space: pre-wrap; word-break: break-word;
  }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.user .bubble { background: var(--user); border-color: var(--user-line); }
  .msg.char .bubble { border-top-left-radius: 4px; }
  .msg.user .bubble { border-top-right-radius: 4px; }
  .stage { font-size: 12px; color: var(--muted); font-style: italic; }
  .extra { font-size: 12px; color: var(--muted); }
  .extra summary { cursor: pointer; outline: none; }
  .extra div { margin-top: 4px; padding-left: 10px; border-left: 2px solid var(--line); }
  .sys { align-self: center; font-size: 12px; color: #c0392b; }

  /* ── 常见问题 ─────────────────────────── */
  #quick {
    display: flex; gap: 8px; flex-wrap: wrap;
    padding: 10px 20px 0; border-top: 1px solid var(--line); background: var(--panel);
  }
  #quick button {
    font-size: 13px; padding: 5px 12px; border-radius: 999px; color: #4a5058;
  }
  footer {
    display: flex; gap: 10px; padding: 12px 20px 16px; background: var(--panel);
  }
  footer textarea {
    flex: 1; resize: none; height: 46px; padding: 12px 14px;
    border-radius: 10px; border: 1px solid var(--line);
    background: #fbfbfc; color: var(--text); font: inherit; outline: none;
  }
  footer textarea:focus { border-color: var(--brand); background: #fff; }
  .typing { color: var(--muted); font-size: 13px; padding: 0 20px 6px; background: var(--panel); }
</style>
</head>
<body>
<header>
  <div class="brandbar"><span class="mark">万</span>华润万家 · 顾客服务</div>
  <div class="deskbar">
    <div class="avatar" id="avatar">?</div>
    <div class="meta">
      <div class="name">
        <span id="charName">加载中…</span>
        <span class="online">在线接待中</span>
      </div>
      <div class="sub" id="charSub">服务台</div>
    </div>
    <div class="chip" id="emotion">状态 —</div>
    <button id="endBtn" style="display:none">结束会话</button>
  </div>
</header>

<section id="setup">
  <div class="welcome">欢迎光临<span>华润万家</span></div>
  <p>我是门店服务台的前台，负责接待、会员积分、退换货、发票和便民服务。<br/>
     请问怎么称呼您？</p>
  <input id="userName" placeholder="您的称呼，例如：张女士" autocomplete="off" />
  <button class="primary" id="startBtn">开始咨询</button>
</section>

<section id="chat">
  <div id="log"></div>
  <div class="typing" id="typing" style="display:none">前台正在处理…</div>
  <div id="quick"></div>
  <footer>
    <textarea id="input" placeholder="请输入您的问题…（Enter 发送，Shift+Enter 换行）"></textarea>
    <button class="primary" id="sendBtn">发送</button>
  </footer>
</section>

<script>
const $ = (id) => document.getElementById(id);
let started = false;
let charName = "前台";

const QUICK_QUESTIONS = [
  "你们几点开门？",
  "会员卡怎么办理？",
  "积分怎么算、怎么查？",
  "买错东西能退吗？",
  "停车怎么收费？",
  "能送货上门吗？",
  "购物卡过期了怎么办？",
  "我想转人工",
];

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
    charName = c.name;
    $("charName").textContent = c.name;
    $("avatar").textContent = c.name.slice(0, 1);
    $("charSub").textContent = c.role_title || (c.profile || "").trim().split("\\n")[0];
  } catch (e) {
    $("charName").textContent = "无法连接服务";
  }
}

function renderQuick() {
  QUICK_QUESTIONS.forEach((q) => {
    const b = document.createElement("button");
    b.textContent = q;
    b.onclick = () => {
      $("input").value = q;
      send();
    };
    $("quick").appendChild(b);
  });
}

function buildExtra(thought, action, emotion) {
  const parts = [];
  if (thought) parts.push("心里想： " + thought);
  if (action) parts.push("动作： " + action);
  if (emotion) parts.push("情绪： " + emotion);
  return parts.join("\\n");
}

function attachExtra(wrap, extra) {
  if (!extra) return;
  const d = document.createElement("details");
  d.className = "extra";
  d.innerHTML = "<summary>查看内心</summary>";
  const inner = document.createElement("div");
  inner.textContent = extra;
  d.appendChild(inner);
  wrap.appendChild(d);
}

function addMessage(role, text, extra, stage) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "user" : "char");
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "您" : charName;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(who, bubble);
  if (stage) {
    const s = document.createElement("div");
    s.className = "stage";
    s.textContent = "（" + stage + "）";
    wrap.appendChild(s);
  }
  attachExtra(wrap, extra);
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
  who.textContent = charName;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrap.append(who, bubble);
  $("log").appendChild(wrap);
  $("log").scrollTop = $("log").scrollHeight;
  return {
    element: wrap,
    append(t) {
      bubble.textContent += t;
      $("log").scrollTop = $("log").scrollHeight;
    },
    setStage(stage) {
      if (!stage) return;
      const s = document.createElement("div");
      s.className = "stage";
      s.textContent = "（" + stage + "）";
      wrap.appendChild(s);
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
    const data = await api("/api/session/start", { user_id: name, user_name: name });
    started = true;
    $("setup").style.display = "none";
    $("chat").style.display = "flex";
    $("endBtn").style.display = "block";
    addMessage(
      "char",
      `您好，${name}。欢迎光临华润万家，我是服务台的前台，您有什么需要都可以直接跟我说。`
    );
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
    if (!gotDelta) stream.append(done.reply || "");
    stream.setStage(done.action);
    const label = done.emotion && done.emotion.label ? done.emotion.label : null;
    attachExtra(stream.element, buildExtra(done.thought, done.action, label));
    (done.system_messages || []).forEach((m) => addSystem(m.message));
    if (label) $("emotion").innerHTML = "状态 <b>" + label + "</b>";
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
  $("endBtn").textContent = "整理中…";
  try {
    const r = await api("/api/session/end");
    addSystem(
      `本次接待结束。情节记忆 +${r.episodic_memories_saved}，语义记忆 +${r.semantic_updates_applied}，关系变化 ${r.relationship_changes}。`
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
renderQuick();
loadCharacter();
</script>
</body>
</html>
"""
