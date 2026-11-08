"""Serve the interactive viewer locally.

    uv run python renderer/serve.py [--port 8765] [--scene-dir DIR]
    then open http://127.0.0.1:8765/renderer/viewer.html  (?net=URL  ?shot=side_noon  ?harness=1)

Mounts: /renderer/ -> this directory, /scene/ -> scene/out (the viewer is opened with ?scene=/scene/).
Network files under renderer/ can be passed as ?net=tests/nets/single_scatter.json (relative to the page).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from harness_client import DEFAULT_SCENE_DIR, start_server  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--scene-dir", default=str(DEFAULT_SCENE_DIR))
a = ap.parse_args()
srv = start_server(Path(a.scene_dir), a.port)
print(f"viewer: http://127.0.0.1:{srv.server_address[1]}/renderer/viewer.html?scene=/scene/   (Ctrl-C to stop)")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    srv.shutdown()
