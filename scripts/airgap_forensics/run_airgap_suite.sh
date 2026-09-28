#!/usr/bin/env bash
# DEFCON ladder for air-gapped testing of transformers.
#   Phase 1 (online staging): fetch test models into a portable cache dir.
#   Phase 2 (simulated air gap): blackhole-socket tests + offline pytest suite.
# Usage: ./run_airgap_suite.sh [CACHE_DIR]     (default CACHE_DIR=/tmp/airgap_cache)
set -euo pipefail

CACHE_DIR="${1:-/tmp/airgap_cache}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

echo "=== DEFCON-0..2: forensic snapshot ==="
"$PY" "$REPO_DIR/scripts/airgap_forensics/forensic_snapshot.py" \
    --output "$CACHE_DIR/forensic_report.json" --repo "$REPO_DIR" || true

echo "=== Phase 1: online staging of tiny test models into $CACHE_DIR ==="
HF_HOME="$CACHE_DIR" "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo in ("hf-internal-testing/tiny-random-BertModel",
             "hf-internal-testing/tiny-random-bert",
             "hf-internal-testing/tiny-random-bert-sharded"):
    try:
        snapshot_download(repo)
        print("STAGED:", repo)
    except Exception as exc:
        print("SKIP (offline already?):", repo, type(exc).__name__)
PYEOF

echo "=== Phase 2a: simulated air-gap end-to-end load (blackhole sockets) ==="
cat > "$CACHE_DIR/airgap_smoke.py" <<'PYEOF'
import socket

class _BlackHole(socket.socket):
    def connect(self, address):
        raise OSError(101, "Network is unreachable (simulated air gap)")

socket.socket = _BlackHole  # any outbound attempt fails like a disconnected NIC

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

mid = "hf-internal-testing/tiny-random-BertModel"
tok = AutoTokenizer.from_pretrained(mid)   # must resolve from cache only
AutoConfig.from_pretrained(mid)
model = AutoModel.from_pretrained(mid)
out = model(**tok("hello air gap", return_tensors="pt"))
assert torch.isfinite(out.last_hidden_state).all()
print("AIRGAPPED SMOKE PASSED:", type(model).__name__, tuple(out.last_hidden_state.shape))
PYEOF
HF_HOME="$CACHE_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PY" "$CACHE_DIR/airgap_smoke.py"

echo "=== Phase 2b: official offline test suite ==="
cd "$REPO_DIR"
HF_HOME="$CACHE_DIR" "$PY" -m pytest tests/utils/test_offline.py -q -p no:libtmux

echo "=== ALL AIR-GAP CHECKS PASSED ==="
