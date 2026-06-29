"""Wrapper: runs exp_chaos.py with a per-API-call timeout so it doesn't hang forever."""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import scripts.exp_chaos as ec

# Monkey-patch client.image_chat with a timed version
_original_image_chat = ec.client.image_chat

def _timed_image_chat(*args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_original_image_chat, *args, **kwargs)
        return fut.result(timeout=60)

ec.client.image_chat = _timed_image_chat

# Now run main
ec.main()
