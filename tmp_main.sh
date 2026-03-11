#!/usr/bin/env bash
# tmp_main.sh - temporary operator launcher for asee backend + Electron viewer

set -u

DEVICE=0
PORT=8765
INTERVAL=45
VOICE=false
CAMERAS=""
CAM_INTERVAL=60
CAPTURE_PROFILE="auto"
CAPTURE_WIDTH=""
CAPTURE_HEIGHT=""
CAPTURE_FPS=""
CAPTURE_FOURCC=""
OPENCV_THREADS=""
AUTO_SHUTDOWN_SEC="0"
DISABLE_FACE_DETECT=false
DETECTION_BACKEND=""
USE_CHROMIUM=false
PWA_INSTALLING=false
OLLAMA_VLM=false
FACE_CAPTURE_DIR="${ASEE_FACE_CAPTURE_DIR:-/home/yuiseki/Workspaces/private/datasets/faces/_raw}"
VIEWER_POLL_INTERVAL_MS="${ASEE_VIEWER_POLL_INTERVAL_MS:-2000}"
VIEWER_RESPAWN="${ASEE_VIEWER_RESPAWN:-1}"
VIEWER_RESPAWN_DELAY_SEC="${ASEE_VIEWER_RESPAWN_DELAY_SEC:-2}"
VIEWER_DISABLE_GPU="${ASEE_VIEWER_DISABLE_GPU:-0}"
VIEWER_USE_GL="${ASEE_VIEWER_USE_GL:-}"
VIEWER_USE_ANGLE="${ASEE_VIEWER_USE_ANGLE:-}"
VIEWER_DISABLE_GPU_SANDBOX="${ASEE_VIEWER_DISABLE_GPU_SANDBOX:-0}"
VIEWER_EXTRA_ARGS="${ASEE_VIEWER_EXTRA_ARGS:-}"
VIEWER_NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-}"
VIEWER_NV_PRIME_RENDER_OFFLOAD_PROVIDER="${__NV_PRIME_RENDER_OFFLOAD_PROVIDER:-}"
VIEWER_GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-}"
VIEWER_DRI_PRIME="${DRI_PRIME:-}"
LAYOUT_MODE=""
LAYOUT_X=0
LAYOUT_Y=1058
LAYOUT_W=2048
LAYOUT_H=1058
export DISPLAY="${DISPLAY:-:0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="${ROOT_DIR}/python"
ELECTRON_DIR="${ROOT_DIR}/electron"
PYTHON_BIN="${ASEE_PYTHON_BIN:-${PYTHON_DIR}/.venv/bin/python}"
COMMAND="${1:-help}"
shift || true

refresh_port_paths() {
  WINDOW_TITLE="ASEE Viewer"
  SERVER_URL="http://127.0.0.1:${PORT}"
  PID_FILE="/tmp/asee_tmp_main_${PORT}.pids"
  SERVER_LOG="/tmp/asee_tmp_main_server_${PORT}.log"
  VIEWER_LOG="/tmp/asee_tmp_main_viewer_${PORT}.log"
  VIEWER_RUNNER="/tmp/asee_tmp_main_viewer_runner_${PORT}.sh"
}

refresh_port_paths

usage() {
  cat <<'EOF'
Usage: ./tmp_main.sh {start|stop|restart|status|layout} [options]

Live capture options:
  --device N
  --port N
  --cameras 0,2,4,6
  --cam-interval N
  --capture-profile auto|720p
  --width N
  --height N
  --fps N
  --fourcc MJPG
  --opencv-threads N
  --auto-shutdown-sec N
  --disable-face-detect
  --detection-backend {opencv,onnxruntime}
  --face-capture-dir PATH

Layout options:
  --full-screen
  --left-bottom
  --frontmost
  --backmost
  --full
  --split-4
  --split-8
  --split-16
  --x N --y N --w N --h N

Legacy compatibility flags accepted as no-op:
  --interval N
  --voice
  --chromium
  --pwa-installing
  --ollama-vlm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --port)
      PORT="$2"
      refresh_port_paths
      shift 2
      ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --voice) VOICE=true; shift ;;
    --cameras) CAMERAS="$2"; shift 2 ;;
    --cam-interval) CAM_INTERVAL="$2"; shift 2 ;;
    --capture-profile) CAPTURE_PROFILE="$2"; shift 2 ;;
    --width) CAPTURE_WIDTH="$2"; shift 2 ;;
    --height) CAPTURE_HEIGHT="$2"; shift 2 ;;
    --fps) CAPTURE_FPS="$2"; shift 2 ;;
    --fourcc) CAPTURE_FOURCC="$2"; shift 2 ;;
    --opencv-threads) OPENCV_THREADS="$2"; shift 2 ;;
    --auto-shutdown-sec) AUTO_SHUTDOWN_SEC="$2"; shift 2 ;;
    --disable-face-detect) DISABLE_FACE_DETECT=true; shift ;;
    --detection-backend) DETECTION_BACKEND="$2"; shift 2 ;;
    --chromium) USE_CHROMIUM=true; shift ;;
    --pwa-installing) PWA_INSTALLING=true; shift ;;
    --ollama-vlm) OLLAMA_VLM=true; shift ;;
    --face-capture-dir) FACE_CAPTURE_DIR="$2"; shift 2 ;;
    --full-screen) LAYOUT_MODE="full-screen"; shift ;;
    --left-bottom) LAYOUT_MODE="left-bottom"; shift ;;
    --frontmost) LAYOUT_MODE="frontmost"; shift ;;
    --backmost) LAYOUT_MODE="backmost"; shift ;;
    --full)
      LAYOUT_X=0
      LAYOUT_Y=0
      LAYOUT_W=4096
      LAYOUT_H=2116
      shift
      ;;
    --split-4)
      LAYOUT_X=0
      LAYOUT_Y=1058
      LAYOUT_W=2048
      LAYOUT_H=1058
      shift
      ;;
    --split-8)
      LAYOUT_X=0
      LAYOUT_Y=1058
      LAYOUT_W=1024
      LAYOUT_H=1058
      shift
      ;;
    --split-16)
      LAYOUT_X=0
      LAYOUT_Y=1587
      LAYOUT_W=1024
      LAYOUT_H=529
      shift
      ;;
    --x) LAYOUT_X="$2"; shift 2 ;;
    --y) LAYOUT_Y="$2"; shift 2 ;;
    --w) LAYOUT_W="$2"; shift 2 ;;
    --h) LAYOUT_H="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

pid_running() {
  local pid="${1:-}"
  local stat
  [[ -n "${pid}" ]] || return 1
  stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  [[ -n "${stat}" ]] || return 1
  [[ "${stat}" != Z* ]]
}

load_pid() {
  local key="$1"
  [[ -f "${PID_FILE}" ]] || return 0
  awk -F= -v key="${key}" '$1 == key { print $2; exit }' "${PID_FILE}" 2>/dev/null
}

save_pids() {
  printf 'server=%s\nviewer=%s\n' "${1:-}" "${2:-}" > "${PID_FILE}"
}

viewer_window_running() {
  command -v wmctrl >/dev/null 2>&1 || return 1
  wmctrl -l 2>/dev/null | grep -Fq "${WINDOW_TITLE}"
}

component_running() {
  local key="$1"
  local pid
  pid="$(load_pid "${key}")"
  if pid_running "${pid}"; then
    return 0
  fi
  if [[ "${key}" == "viewer" ]] && viewer_window_running; then
    return 0
  fi
  return 1
}

wait_for_server() {
  command -v curl >/dev/null 2>&1 || return 0
  echo -n "Waiting for server ${SERVER_URL}"
  local attempt
  for attempt in $(seq 1 20); do
    if curl -sf "${SERVER_URL}/status" 2>/dev/null | grep -Eq '"running"[[:space:]]*:[[:space:]]*true'; then
      echo " ready (${attempt}s)"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo " timeout"
  return 1
}

wait_for_window() {
  command -v wmctrl >/dev/null 2>&1 || return 0
  local attempt
  for attempt in $(seq 1 20); do
    if viewer_window_running; then
      return 0
    fi
    sleep 1
  done
  return 1
}

close_viewer_window() {
  command -v wmctrl >/dev/null 2>&1 || return 0
  wmctrl -c "${WINDOW_TITLE}" >/dev/null 2>&1 || true
}

kill_process_group() {
  local key="$1"
  local pid pgid deadline
  pid="$(load_pid "${key}")"
  if [[ -z "${pid}" ]]; then
    echo "  ${key}: no pid recorded"
    return 0
  fi
  if ! pid_running "${pid}"; then
    echo "  ${key}: not running"
    return 0
  fi

  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${key}" == "viewer" ]]; then
    close_viewer_window
    sleep 1
  fi

  if [[ -n "${pgid}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi

  deadline=$((SECONDS + 5))
  while pid_running "${pid}" && (( SECONDS < deadline )); do
    sleep 0.2
  done

  if pid_running "${pid}"; then
    if [[ -n "${pgid}" ]]; then
      kill -KILL -- "-${pgid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    else
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi

  deadline=$((SECONDS + 2))
  while pid_running "${pid}" && (( SECONDS < deadline )); do
    sleep 0.1
  done

  if pid_running "${pid}"; then
    echo "  ${key}: still running (pid=${pid})"
    return 1
  fi

  echo "  ${key}: stopped"
  return 0
}

apply_kwin_script() {
  local script_path="$1"
  command -v qdbus >/dev/null 2>&1 || return 0

  local plugin
  plugin="asee_tmp_main_${PORT}_$(date +%s)"
  DISPLAY="${DISPLAY}" qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "${script_path}" "${plugin}" >/dev/null 2>&1 || return 0
  DISPLAY="${DISPLAY}" qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.start >/dev/null 2>&1 || true
  sleep 0.5
  DISPLAY="${DISPLAY}" qdbus org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "${plugin}" >/dev/null 2>&1 || true
}

apply_layout_mode() {
  local js_file="/tmp/asee_tmp_main_layout_${PORT}.js"

  case "${LAYOUT_MODE}" in
    full-screen)
      cat > "${js_file}" <<EOF
var clients = workspace.clientList();
for (var i = 0; i < clients.length; i++) {
    var c = clients[i];
    if (c.caption.indexOf("${WINDOW_TITLE}") !== -1) {
        c.noBorder = true;
        c.keepAbove = true;
        c.onAllDesktops = true;
        c.fullScreen = true;
        break;
    }
}
EOF
      ;;
    frontmost)
      cat > "${js_file}" <<EOF
var clients = workspace.clientList();
for (var i = 0; i < clients.length; i++) {
    var c = clients[i];
    if (c.caption.indexOf("${WINDOW_TITLE}") !== -1) {
        c.keepBelow = false;
        c.keepAbove = true;
        break;
    }
}
EOF
      ;;
    backmost)
      cat > "${js_file}" <<EOF
var clients = workspace.clientList();
for (var i = 0; i < clients.length; i++) {
    var c = clients[i];
    if (c.caption.indexOf("${WINDOW_TITLE}") !== -1) {
        c.keepAbove = false;
        c.keepBelow = true;
        break;
    }
}
EOF
      ;;
    *)
      cat > "${js_file}" <<EOF
var clients = workspace.clientList();
for (var i = 0; i < clients.length; i++) {
    var c = clients[i];
    if (c.caption.indexOf("${WINDOW_TITLE}") !== -1) {
        c.fullScreen = false;
        c.noBorder = true;
        c.keepAbove = true;
        c.onAllDesktops = true;
        var g = c.frameGeometry;
        g.x = ${LAYOUT_X};
        g.y = ${LAYOUT_Y};
        g.width = ${LAYOUT_W};
        g.height = ${LAYOUT_H};
        c.frameGeometry = g;
        break;
    }
}
EOF
      ;;
  esac

  apply_kwin_script "${js_file}"
  rm -f "${js_file}"
}

align_default_window() {
  LAYOUT_MODE="left-bottom"
  wait_for_window || return 0
  apply_layout_mode
}

require_start_prereqs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python runtime: ${PYTHON_BIN}" >&2
    return 1
  fi
  if [[ ! -d "${ELECTRON_DIR}" ]]; then
    echo "Missing Electron project: ${ELECTRON_DIR}" >&2
    return 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "Missing npm in PATH" >&2
    return 1
  fi
  if ! command -v node >/dev/null 2>&1; then
    echo "Missing node in PATH" >&2
    return 1
  fi
}

build_viewer() {
  printf '[%s] viewer build start\n' "$(date --iso-8601=seconds)" >> "${VIEWER_LOG}"
  (
    cd "${ELECTRON_DIR}" || exit 1
    env DISPLAY="${DISPLAY}" npm run build
  ) >> "${VIEWER_LOG}" 2>&1
}

write_viewer_runner() {
  local disable_gpu_arg=""
  if [[ "${VIEWER_DISABLE_GPU}" == "1" ]]; then
    disable_gpu_arg="--disable-gpu"
  fi

  cat > "${VIEWER_RUNNER}" <<EOF
#!/usr/bin/env bash
set -u
trap 'exit 0' TERM INT HUP
cd "${ELECTRON_DIR}" || exit 1

while true; do
  printf '[%s] viewer run start\n' "\$(date --iso-8601=seconds)" >> "${VIEWER_LOG}"
  printf '[%s] viewer gpu env disable_gpu=%s use_gl=%s use_angle=%s disable_gpu_sandbox=%s extra_args=%s prime_offload=%s prime_provider=%s glx_vendor=%s dri_prime=%s\n' "\$(date --iso-8601=seconds)" "${VIEWER_DISABLE_GPU}" "${VIEWER_USE_GL}" "${VIEWER_USE_ANGLE}" "${VIEWER_DISABLE_GPU_SANDBOX}" "${VIEWER_EXTRA_ARGS}" "${VIEWER_NV_PRIME_RENDER_OFFLOAD}" "${VIEWER_NV_PRIME_RENDER_OFFLOAD_PROVIDER}" "${VIEWER_GLX_VENDOR_LIBRARY_NAME}" "${VIEWER_DRI_PRIME}" >> "${VIEWER_LOG}"
  env \\
    DISPLAY="${DISPLAY}" \\
    ASEE_VIEWER_BACKEND_URL="${SERVER_URL}" \\
    ASEE_VIEWER_TITLE="${WINDOW_TITLE}" \\
    ASEE_VIEWER_POLL_INTERVAL_MS="${VIEWER_POLL_INTERVAL_MS}" \\
    ASEE_VIEWER_DISABLE_GPU="${VIEWER_DISABLE_GPU}" \\
    ASEE_VIEWER_USE_GL="${VIEWER_USE_GL}" \\
    ASEE_VIEWER_USE_ANGLE="${VIEWER_USE_ANGLE}" \\
    ASEE_VIEWER_DISABLE_GPU_SANDBOX="${VIEWER_DISABLE_GPU_SANDBOX}" \\
    ASEE_VIEWER_EXTRA_ARGS="${VIEWER_EXTRA_ARGS}" \\
    __NV_PRIME_RENDER_OFFLOAD="${VIEWER_NV_PRIME_RENDER_OFFLOAD}" \\
    __NV_PRIME_RENDER_OFFLOAD_PROVIDER="${VIEWER_NV_PRIME_RENDER_OFFLOAD_PROVIDER}" \\
    __GLX_VENDOR_LIBRARY_NAME="${VIEWER_GLX_VENDOR_LIBRARY_NAME}" \\
    DRI_PRIME="${VIEWER_DRI_PRIME}" \\
    node ./scripts/run-electron-with-x11-env.mjs --skip-build ${disable_gpu_arg} >> "${VIEWER_LOG}" 2>&1 &
  viewer_child=\$!
  for attempt in \$(seq 1 20); do
    if command -v wmctrl >/dev/null 2>&1 && DISPLAY="${DISPLAY}" wmctrl -l 2>/dev/null | grep -Fq "${WINDOW_TITLE}"; then
      "${ROOT_DIR}/tmp_main.sh" layout --port "${PORT}" --left-bottom >> "${VIEWER_LOG}" 2>&1 || true
      break
    fi
    if ! kill -0 "\${viewer_child}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  wait "\${viewer_child}"
  viewer_rc=\$?
  printf '[%s] viewer run exit code=%s\n' "\$(date --iso-8601=seconds)" "\${viewer_rc}" >> "${VIEWER_LOG}"
  if [[ "${VIEWER_RESPAWN}" != "1" ]]; then
    exit "\${viewer_rc}"
  fi
  printf '[%s] viewer respawn after %ss\n' "\$(date --iso-8601=seconds)" "${VIEWER_RESPAWN_DELAY_SEC}" >> "${VIEWER_LOG}"
  sleep "${VIEWER_RESPAWN_DELAY_SEC}"
done
EOF
  chmod +x "${VIEWER_RUNNER}"
}

launch_viewer_supervisor() {
  write_viewer_runner
  setsid "${VIEWER_RUNNER}" > /dev/null 2>&1 &
  echo "$!"
}

cmd_start() {
  require_start_prereqs || exit 1

  local server_running viewer_running
  server_running=false
  viewer_running=false
  component_running server && server_running=true
  component_running viewer && viewer_running=true

  if [[ "${server_running}" == "true" && "${viewer_running}" == "true" ]]; then
    echo "ASEE tmp main on port ${PORT} is already active."
    exit 0
  fi

  if [[ "${server_running}" == "true" || "${viewer_running}" == "true" || -f "${PID_FILE}" ]]; then
    echo "Cleaning previous partial state on port ${PORT}"
    cmd_stop >/dev/null 2>&1 || true
  fi

  echo "=== ASEE TMP MAIN START (PORT ${PORT}) ==="
  printf '[%s] tmp_main start\n' "$(date --iso-8601=seconds)" >> "${VIEWER_LOG}"

  local -a server_args
  server_args=(
    -m
    asee.video_server
    --port "${PORT}"
    --device "${DEVICE}"
    --allow-live-camera
    --cam-interval "${CAM_INTERVAL}"
    --title "${WINDOW_TITLE}"
    --face-capture-dir "${FACE_CAPTURE_DIR}"
  )
  [[ -n "${CAMERAS}" ]] && server_args+=(--cameras "${CAMERAS}")
  [[ "${CAPTURE_PROFILE}" != "auto" ]] && server_args+=(--capture-profile "${CAPTURE_PROFILE}")
  [[ -n "${CAPTURE_WIDTH}" ]] && server_args+=(--width "${CAPTURE_WIDTH}")
  [[ -n "${CAPTURE_HEIGHT}" ]] && server_args+=(--height "${CAPTURE_HEIGHT}")
  [[ -n "${CAPTURE_FPS}" ]] && server_args+=(--fps "${CAPTURE_FPS}")
  [[ -n "${CAPTURE_FOURCC}" ]] && server_args+=(--fourcc "${CAPTURE_FOURCC}")
  [[ -n "${OPENCV_THREADS}" ]] && server_args+=(--opencv-threads "${OPENCV_THREADS}")
  [[ "${AUTO_SHUTDOWN_SEC}" != "0" ]] && server_args+=(--auto-shutdown-sec "${AUTO_SHUTDOWN_SEC}")
  [[ "${DISABLE_FACE_DETECT}" == "true" ]] && server_args+=(--disable-face-detect)
  [[ -n "${DETECTION_BACKEND}" ]] && server_args+=(--detection-backend "${DETECTION_BACKEND}")

  env DISPLAY="${DISPLAY}" setsid "${PYTHON_BIN}" "${server_args[@]}" > "${SERVER_LOG}" 2>&1 < /dev/null &
  local server_pid="$!"
  echo "  backend: pid=${server_pid}"

  if ! wait_for_server; then
    echo "Backend did not become ready; cleaning up." >&2
    save_pids "${server_pid}" ""
    cmd_stop >/dev/null 2>&1 || true
    exit 1
  fi

  local viewer_pid
  if ! build_viewer; then
    echo "Viewer build failed; cleaning up." >&2
    save_pids "${server_pid}" ""
    cmd_stop >/dev/null 2>&1 || true
    exit 1
  fi
  viewer_pid="$(launch_viewer_supervisor)"

  save_pids "${server_pid}" "${viewer_pid}"
  echo "  viewer: pid=${viewer_pid:-unknown}"
  align_default_window || true

  if [[ "${USE_CHROMIUM}" == "true" || "${PWA_INSTALLING}" == "true" || "${VOICE}" == "true" || "${OLLAMA_VLM}" == "true" ]]; then
    echo "  note: legacy compatibility flags were ignored"
  fi
  echo "=== ASEE TMP MAIN ${PORT} ACTIVE ==="
}

cmd_stop() {
  echo "=== ASEE TMP MAIN STOP (PORT ${PORT}) ==="
  local rc=0
  kill_process_group viewer || rc=1
  kill_process_group server || rc=1
  rm -f "${PID_FILE}"
  rm -f "${VIEWER_RUNNER}"
  echo "=== ASEE TMP MAIN ${PORT} STOPPED ==="
  return "${rc}"
}

cmd_status() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "ASEE tmp main (PORT ${PORT}): STOPPED (no pid file)"
    return 0
  fi

  local all_running=true
  local key pid
  for key in server viewer; do
    pid="$(load_pid "${key}")"
    if component_running "${key}"; then
      echo "  ${key}: RUNNING (pid=${pid:-unknown})"
    else
      echo "  ${key}: STOPPED"
      all_running=false
    fi
  done

  if [[ "${all_running}" == "true" ]]; then
    echo "ASEE tmp main (PORT ${PORT}): ACTIVE"
  else
    echo "ASEE tmp main (PORT ${PORT}): PARTIAL"
  fi
}

cmd_layout() {
  if ! component_running viewer; then
    echo "ASEE tmp main (PORT ${PORT}): viewer not running" >&2
    exit 1
  fi
  if [[ -z "${LAYOUT_MODE}" ]]; then
    echo "Usage: ./tmp_main.sh layout {--full-screen|--left-bottom|--frontmost|--backmost} [--port N]" >&2
    exit 1
  fi
  apply_layout_mode
  echo "  Layout applied: ${LAYOUT_MODE}"
}

case "${COMMAND}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart)
    cmd_stop >/dev/null 2>&1 || true
    sleep 1
    cmd_start
    ;;
  status) cmd_status ;;
  layout) cmd_layout ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
