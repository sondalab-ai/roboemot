#!/bin/bash
set -e
S="${WAKEWORD_DIR:?set WAKEWORD_DIR to a scratch folder outside the repo}"
REV=e7229f7d33a10e145e109b886d661984201ac3c4
D=$S/data/piper; mkdir -p $D
fetch() { # lang_dir voice quality
  local name="$1"; local lang=${name%%-*}; local fam=${lang%%_*}
  local rest=${name#*-}; local spk=${rest%-*}; local q=${rest##*-}
  for ext in onnx onnx.json; do
    [ -s "$D/$name.$ext" ] || curl -sfL -o "$D/$name.$ext" "https://huggingface.co/rhasspy/piper-voices/resolve/$REV/$fam/$lang/$spk/$q/$name.$ext"
  done
}
for v in it_IT-riccardo-x_low it_IT-serena-high it_IT-serena-medium it_IT-paola-medium en_US-libritts_r-medium es_ES-davefx-medium es_MX-claude-high pt_BR-faber-medium pt_PT-tugão-medium ro_RO-mihai-medium es_AR-daniela-high pt_BR-cadu-medium; do
  fetch "$v"
done
ls -la $D
du -sh $D
