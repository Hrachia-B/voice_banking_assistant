from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livekit.api import AccessToken, VideoGrants

from app.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LiveKit access token for local voice testing.")
    parser.add_argument("--room", default="bank-agent-room", help="Room name.")
    parser.add_argument("--identity", default="voice-tester", help="Participant identity.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise SystemExit("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env.")

    # LiveKit local dev mode uses the official devkey/secret pair.
    warnings.filterwarnings("ignore", message="The HMAC key is 6 bytes long.*")

    token = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(args.identity)
        .with_name(args.identity)
        .with_grants(VideoGrants(room_join=True, room=args.room))
        .to_jwt()
    )
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
