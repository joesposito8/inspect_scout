"""Links to messages in Scout View."""

import base64
import os
from urllib.parse import quote, urlencode

from inspect_ai._util.file import absolute_file_path


def view_url(
    transcripts: str,
    transcript_id: str,
    message_id: str,
    *,
    base_url: str | None = None,
) -> str:
    """Build a message link using SCOUT_VIEW_URL or the standard local viewer."""
    encoded = (
        base64.urlsafe_b64encode(absolute_file_path(transcripts).encode())
        .decode()
        .rstrip("=")
    )
    base = (
        base_url
        if base_url is not None
        else os.environ.get("SCOUT_VIEW_URL", "http://localhost:7576")
    )
    query = urlencode({"tab": "transcript-messages", "message": message_id})
    return f"{base.rstrip('/')}/#/transcripts/{encoded}/{quote(transcript_id, safe='')}?{query}"
