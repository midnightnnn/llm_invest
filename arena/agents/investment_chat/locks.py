from __future__ import annotations

import threading
from collections import defaultdict

_TENANT_WRITE_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)
_TENANT_LOCKS_GUARD = threading.Lock()


def tenant_lock(tenant_id: str) -> threading.RLock:
    with _TENANT_LOCKS_GUARD:
        return _TENANT_WRITE_LOCKS[tenant_id]
