#!/usr/bin/env bash
# Physical acceptance for S25 runtime activation steering.
#
# Runs one prompt through Balanced / Concise / Detailed on the real device via
# the normal Brain gRPC path and prints, for each, the artifact that answered,
# the steering fields that were actually sent, and the per-phase aux-tensor
# write counts the device reported. Balanced is expected to stay on the stock
# GenieX Base artifact; only Concise/Detailed should reach the forked runtime.
set -uo pipefail

BRAIN="${BRAIN:-127.0.0.1:50051}"
DASHBOARD="${DASHBOARD:-http://127.0.0.1:8080}"
SERIAL="${SERIAL:-R3CXC0805HW}"
PROMPT="${PROMPT:-Explain why the sky is blue.}"
export MSYS_NO_PATHCONV=1

cd "$(dirname "$0")/.."

# Discovered, not hardcoded: re-enrolling the phone can mint a new device_id,
# and pinning a stale one would silently route the acceptance run somewhere
# other than the device under test.
DEVICE="${DEVICE:-$(curl -s --max-time 5 "$DASHBOARD/api/devices" | python -c "
import sys, json
ids = [d['device_id'] for d in json.load(sys.stdin)
       if d.get('device_id','').startswith('android') and d.get('connected')]
print(ids[0] if ids else '')
")}"
if [ -z "$DEVICE" ]; then
  echo "No connected Android device is registered with Brain at $DASHBOARD." >&2
  echo "Enrol the phone (PersonaCare -> Connect) and re-run." >&2
  exit 2
fi
echo "device: $DEVICE"

adb -s "$SERIAL" logcat -c 2>/dev/null || true

for persona in balanced concise detailed; do
  echo
  echo "==================== $persona ===================="
  python scripts/submit_task.py "$PROMPT" \
      --brain "$BRAIN" \
      --preferred-mode local \
      --origin-device-id "$DEVICE" \
      --persona-id "$persona" \
      --timeout-ms 90000
done

echo
echo "==================== device aux evidence ===================="
# Only the forked bridge logs these; a Balanced request must not appear here.
adb -s "$SERIAL" logcat -d -s "DragonNestGenieXAux:*" 2>/dev/null \
  | grep -E "prefill_aux_writes|loaded steering context" || echo "(no aux activity logged)"
