#!/usr/bin/env bash
# Run the Sondalab design audit (sl-audit) on the web UI's own files. Local only: needs Node.
# The vendored kit under static/vendor/ is the token source and is not audited.
#   scripts/audit_webui.sh                  # npx fetches the pinned @sondalab/ui-kit
#   SL_AUDIT=/path/to/agent/sl-audit.mjs scripts/audit_webui.sh   # use an unpacked kit instead
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(python3 -c "import json,pathlib;print(json.loads(pathlib.Path('packages/roboemot-core/src/roboemot/webui/static/vendor/sondalab-ui-kit/VENDORED.json').read_text())['version'])")
SURFACE="packages/roboemot-core/src/roboemot/webui/static/*.{html,css,js}"
ARGS=(--surface "$SURFACE" --product roboemot --substrate App)
if [ -n "${SL_AUDIT:-}" ]; then
  node "$SL_AUDIT" "${ARGS[@]}"
else
  npx --yes -p "@sondalab/ui-kit@${VERSION}" sl-audit "${ARGS[@]}"
fi
