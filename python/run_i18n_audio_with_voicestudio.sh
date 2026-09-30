#!/bin/bash
set -euo pipefail

UV_BIN="$1"
SCRIPT_PATH="$2"
PROJECT_DIR="$3"
BACKEND_PLIST="$4"
BACKEND_LABEL="com.picturetrip.voicestudio-backend"
DOMAIN="gui/$(id -u)"
BACKEND_TARGET="$DOMAIN/$BACKEND_LABEL"

cd "$PROJECT_DIR"

pending_output="$("$UV_BIN" run --no-project --default-index https://pypi.org/simple \
  --with pymysql -- python "$SCRIPT_PATH" --check-pending --gender female)"
pending_count="$(printf '%s\n' "$pending_output" \
  | /usr/bin/sed -n 's/^PENDING_AUDIO_COUNT=//p' \
  | /usr/bin/tail -n 1)"
if [[ ! "$pending_count" =~ ^[0-9]+$ ]]; then
  echo "Pending audio check returned an invalid result: $pending_output" >&2
  exit 1
fi
if (( pending_count == 0 )); then
  echo "No pending audio; VoiceStudio was not started."
  exit 0
fi

model_status_ready() {
  /usr/bin/curl --noproxy '*' --silent --show-error --fail \
    --connect-timeout 2 --max-time 3 \
    http://127.0.0.1:3900/model/status >/dev/null 2>&1
}

if ! model_status_ready; then
  if [[ ! -f "$BACKEND_PLIST" ]]; then
    echo "VoiceStudio is unavailable and backend LaunchAgent is missing: $BACKEND_PLIST" >&2
    exit 1
  fi

  if ! launchctl print "$BACKEND_TARGET" >/dev/null 2>&1; then
    if ! launchctl bootstrap "$DOMAIN" "$BACKEND_PLIST" \
      && ! launchctl print "$BACKEND_TARGET" >/dev/null 2>&1; then
      echo "Could not load the VoiceStudio backend LaunchAgent." >&2
      exit 1
    fi
  fi
  launchctl kickstart "$BACKEND_TARGET"

  ready=false
  deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if model_status_ready; then
      ready=true
      echo "VoiceStudio API is ready."
      break
    fi
    sleep 2
  done
  if [[ "$ready" != true ]]; then
    echo "VoiceStudio API did not become ready within 180 seconds; this audio run was skipped." >&2
    exit 1
  fi
else
  echo "VoiceStudio API is already ready."
fi

exec "$UV_BIN" run --no-project --default-index https://pypi.org/simple \
  --with pymysql -- python "$SCRIPT_PATH" --apply --skip-asr --gender female
