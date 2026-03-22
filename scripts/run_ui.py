from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.ui.gradio_app import create_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local UI for the Armenian bank assistant.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the UI server.")
    parser.add_argument(
        "--port",
        type=int,
        help="Optional port to bind the UI server. If omitted, Gradio chooses an available one.",
    )
    parser.add_argument("--index-dir", help="Path to a retrieval index directory.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio share link. Off by default for local usage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    index_dir = Path(args.index_dir) if args.index_dir else None
    demo = create_demo(settings, index_dir=index_dir)
    allowed_paths = [str((settings.project_root / "tmp").resolve())]
    launch_kwargs = {
        "server_name": args.host,
        "share": args.share,
        "allowed_paths": allowed_paths,
        "inbrowser": False,
    }
    if args.port is not None:
        launch_kwargs["server_port"] = args.port

    demo.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
