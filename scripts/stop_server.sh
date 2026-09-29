#!/usr/bin/env bash
#
# Pneuma Core —— 一键停止服务
#
# 停止顺序：
#   1. HTTP 优雅关闭（POST /api/admin/shutdown，仅本机可达）
#   2. PID 文件（vault/pneuma.pid）
#   3. 命令行匹配（python -m pneuma_core.server）
#   4. 端口占用（兜底）
#
# 用法：
#   ./scripts/stop_server.sh
#   ./scripts/stop_server.sh --quiet     # 静默模式（供 start 脚本内部调用）
#   ./scripts/stop_server.sh --port 8010 # 指定端口（覆盖 .env）
#

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUIET=0
PORT_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quiet) QUIET=1 ;;
    --port)  PORT_OVERRIDE="${2:-}"; shift ;;
    *) ;;
  esac
  shift
done

info() { [ "$QUIET" = "1" ] || printf '\033[36m[INFO]\033[0m  %s\n' "$*"; }
warn() { [ "$QUIET" = "1" ] || printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }

# 读取配置（.env 不存在时用默认值）
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

PORT="${PORT_OVERRIDE:-${PNEUMA_PORT:-8001}}"
RUN_DIR="$(dirname "${PNEUMA_DB_PATH:-vault/pneuma.db}")"
PID_FILE="$RUN_DIR/pneuma.pid"

port_free() { ! ss -ltn 2>/dev/null | grep -q ":${1}\b"; }

wait_stopped() {
  local tries=30
  while [ "$tries" -gt 0 ]; do
    port_free "$PORT" && return 0
    sleep 0.5
    tries=$((tries - 1))
  done
  return 1
}

# ── 1) HTTP 优雅关闭 ─────────────────────────────────────────
if ! port_free "$PORT"; then
  code="$(curl -sS -m 5 -o /dev/null -w '%{http_code}' \
          -X POST "http://127.0.0.1:${PORT}/api/admin/shutdown" 2>/dev/null || true)"
  if [ "$code" = "200" ]; then
    info "已发送关闭请求，等待退出…"
    if wait_stopped; then
      rm -f "$PID_FILE"
      info "服务已优雅停止（端口 $PORT 已释放）"
      exit 0
    fi
    warn "优雅关闭超时，继续清理残留进程"
  fi
fi

# ── 2) 按 PID 文件 ───────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    info "停止进程 (PID $PID)…"
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

# ── 3) 按命令行匹配 ──────────────────────────────────────────
PIDS="$(pgrep -f '[p]neuma_core\.server' 2>/dev/null || true)"
if [ -n "$PIDS" ]; then
  info "清理残留进程: $(echo "$PIDS" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  sleep 1
  PIDS="$(pgrep -f '[p]neuma_core\.server' 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$PIDS" ] && kill -9 $PIDS 2>/dev/null || true
fi

# ── 4) 按端口占用（兜底）────────────────────────────────────
if ! port_free "$PORT"; then
  warn "端口 $PORT 仍被占用，尝试强制释放…"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -n tcp "$PORT" >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$PORT" 2>/dev/null | xargs -r kill 2>/dev/null || true
  fi
  sleep 1
fi

# ── 结果 ─────────────────────────────────────────────────────
if port_free "$PORT"; then
  info "服务已停止（端口 $PORT 已释放）"
  exit 0
fi

warn "端口 $PORT 仍被占用。可能是其他程序占用，或当前环境不允许跨会话结束进程。"
exit 1
