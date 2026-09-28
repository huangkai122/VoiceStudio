#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS launchd agents." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/generate_i18n_audio.py"
RUNNER_PATH="$SCRIPT_DIR/run_i18n_audio_with_voicestudio.sh"
UV_BIN="$(command -v uv || true)"
PYTHON_BIN="$(command -v python3 || true)"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Missing generator: $SCRIPT_PATH" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
  echo "Missing configuration: $SCRIPT_DIR/.env" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "Missing generation runner: $RUNNER_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/backend/main.py" ]]; then
  echo "Missing VoiceStudio backend: $PROJECT_DIR/backend/main.py" >&2
  echo "Install this LaunchAgent from VoiceStudio/python." >&2
  exit 1
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv was not found in PATH. Install uv, then rerun this installer." >&2
  exit 1
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 was not found in PATH." >&2
  exit 1
fi

LABEL="com.picturetrip.generate-i18n-audio"
BACKEND_LABEL="com.picturetrip.voicestudio-backend"
AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/PictureTrip"
PLIST_PATH="$AGENT_DIR/$LABEL.plist"
BACKEND_PLIST_PATH="$AGENT_DIR/$BACKEND_LABEL.plist"
OUT_LOG="$LOG_DIR/generate-i18n-audio.log"
ERR_LOG="$LOG_DIR/generate-i18n-audio.error.log"
BACKEND_OUT_LOG="$LOG_DIR/voicestudio-backend.log"
BACKEND_ERR_LOG="$LOG_DIR/voicestudio-backend.error.log"

mkdir -p "$AGENT_DIR" "$LOG_DIR"

"$PYTHON_BIN" - \
  "$PLIST_PATH" "$BACKEND_PLIST_PATH" "$UV_BIN" "$SCRIPT_PATH" "$RUNNER_PATH" \
  "$PROJECT_DIR" "$OUT_LOG" "$ERR_LOG" "$BACKEND_OUT_LOG" "$BACKEND_ERR_LOG" <<'PYTHON'
import plistlib
import sys

(
    plist_path,
    backend_plist_path,
    uv_path,
    script_path,
    runner_path,
    working_dir,
    out_log,
    err_log,
    backend_out_log,
    backend_err_log,
) = sys.argv[1:]
payload = {
    "Label": "com.picturetrip.generate-i18n-audio",
    "ProgramArguments": [
        "/bin/bash",
        runner_path,
        uv_path,
        script_path,
        working_dir,
        backend_plist_path,
    ],
    "WorkingDirectory": working_dir,
    "StartInterval": 300,
    "RunAtLoad": True,
    "EnvironmentVariables": {
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    },
    "StandardOutPath": out_log,
    "StandardErrorPath": err_log,
}
backend_payload = {
    "Label": "com.picturetrip.voicestudio-backend",
    "ProgramArguments": [uv_path, "run", "python", "backend/main.py"],
    "WorkingDirectory": working_dir,
    "StandardOutPath": backend_out_log,
    "StandardErrorPath": backend_err_log,
    "EnvironmentVariables": {
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    },
}
for path, data in ((plist_path, payload), (backend_plist_path, backend_payload)):
    with open(path, "wb") as plist_file:
        plistlib.dump(data, plist_file, fmt=plistlib.FMT_XML, sort_keys=False)
PYTHON

plutil -lint "$PLIST_PATH"
plutil -lint "$BACKEND_PLIST_PATH"

DOMAIN="gui/$(id -u)"
if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL"
fi
launchctl bootstrap "$DOMAIN" "$PLIST_PATH"

echo "Installed $LABEL; it runs once now and then every 5 minutes."
echo "VoiceStudio backend starts on demand and stays running after the first generation."
echo "Logs: $OUT_LOG"
echo "Errors: $ERR_LOG"
echo "VoiceStudio logs: $BACKEND_OUT_LOG"
echo "VoiceStudio errors: $BACKEND_ERR_LOG"
