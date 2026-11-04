"""Map every generated clip ('<set>/<file>.wav') to the text it was synthesised from.

Usage (piper venv): python -I textmap.py OUT.json
gen_clips.jobs() is deterministic (seeded), so the map matches the clips gen_clips.py wrote.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gen_clips  # noqa: E402

jobs = gen_clips.jobs("X")
Path(sys.argv[1]).write_text(
    json.dumps({"/".join(j[0].split("/")[-2:]): j[3] for j in jobs}, ensure_ascii=False)
)
print(len(jobs), "clips")
