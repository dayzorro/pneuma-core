#!/usr/bin/env bash
#
# Pneuma Core —— 一键启动服务（后台常驻）
#
#   1. 先调用 stop_server.sh 清理残留进程
#   2. 加载 .env
#   3. 用 nohup 后台启动，写入 PID 文件与日志
#   4. 轮询 /healthz 确认启动成功，并打印访问地址
#
# 用法：
#   ./scripts/start_server.sh              # 后台启动（默认）
#   ./scripts/start_server.sh --fg         # 前台启动（调试用，Ctrl+C 停止）
#   ./scripts/start_server.sh --port 8010  # 指定端口（覆盖 .env）
#
# 相关文件（位于 PNEUMA_DB_PATH 所在目录，默认 vault/）：
#   vault/pneuma.pid    进程 PID
#   vault/server.log    运行日志
#

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
ENV_FILE="$ROOT/.env"
FOREGROUND=0
PORT_OVERRIDE=""
HOST_OVERRIDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --fg)   FOREGROUND=1 ;;
    --port) PORT_OVERRIDE="${2:-}"; shift ;;
    --host) HOST_OVERRIDE="${2:-}"; shift ;;
    *) ;;
  esac
  shift
done

info()  { printf '\033[36m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[31m[ERROR]\033[0m %s\n' "$*" >&2; }

# ── 前置检查 ─────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
  error "未找到 .env，请先执行："
  echo "         cp .env.example .env   # 然后填入你的 API 配置"
  exit 1
fi

if [ ! -x "$PYTHON" ]; then
  error "未找到虚拟环境解释器：$PYTHON"
  echo "         请先执行："
  echo "           uv venv .venv"
  echo "           uv pip install --python .venv -e \".[server]\""
  exit 1
fi

# ── 加载配置（.env 之后应用命令行覆盖）──────────────────────
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

[ -n "$PORT_OVERRIDE" ] && PNEUMA_PORT="$PORT_OVERRIDE"
[ -n "$HOST_OVERRIDE" ] && PNEUMA_HOST="$HOST_OVERRIDE"
export PNEUMA_PORT PNEUMA_HOST

PORT="${PNEUMA_PORT:-8001}"
HOST="${PNEUMA_HOST:-0.0.0.0}"
RUN_DIR="$(dirname "${PNEUMA_DB_PATH:-vault/pneuma.db}")"
PID_FILE="$RUN_DIR/pneuma.pid"
LOG_FILE="$RUN_DIR/server.log"
mkdir -p "$RUN_DIR"

# ── 清理旧进程 ───────────────────────────────────────────────
"$ROOT/scripts/stop_server.sh" --quiet --port "$PORT" || true

# ── 信息 ─────────────────────────────────────────────────────
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
info "角色卡 : ${PNEUMA_CHARACTER_FILE:-examples/aine.character.yaml}"
info "模型   : ${PNEUMA_LLM_MODEL:-<默认>}"
info "监听   : ${HOST}:${PORT}"
echo

# ── 前台模式 ─────────────────────────────────────────────────
if [ "$FOREGROUND" = "1" ]; then
  info "前台启动（Ctrl+C 停止）"
  echo
  exec "$PYTHON" -m pneuma_core.server
fi

# ── 后台模式 ─────────────────────────────────────────────────
info "后台启动，日志: $LOG_FILE"
nohup "$PYTHON" -m pneuma_core.server >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

# 轮询健康检查（最多 ~20 秒）
OK=0
for _ in $(seq 1 40); do
  kill -0 "$PID" 2>/dev/null || break          # 进程已退出 → 启动失败
  if curl -sS -m 2 -o /dev/null "http://127.0.0.1:${PORT}/healthz" 2>/dev/null; then
    OK=1
    break
  fi
  sleep 0.5
done

if [ "$OK" != "1" ]; then
  error "启动失败，日志末尾："
  tail -n 25 "$LOG_FILE" 2>/dev/null || true
  rm -f "$PID_FILE"
  exit 1
fi

echo
info "启动成功 (PID $PID)"
echo
printf '        \033[1m本机访问  \033[0m http://localhost:%s/\n' "$PORT"
[ -n "$LAN_IP" ] && printf '        \033[1m局域网访问\033[0m http://%s:%s/\n' "$LAN_IP" "$PORT"
printf '        \033[1mAPI 文档  \033[0m http://localhost:%s/docs\n' "$PORT"
echo
info "查看日志: tail -f $LOG_FILE"
info "停止服务: ./scripts/stop_server.sh"
