"""
Pairing endpoints for InterviewSimXR.

Drop this file into your backend project (next to wherever your existing
/interview routes live) and include the router in your main FastAPI app:

    from pairing import router as pairing_router
    app.include_router(pairing_router, prefix="/interview")

The three endpoints:
  POST /interview/new-pairing            -> Quest calls this on scene load
  POST /interview/pair                   -> Frontend calls this after upload
  GET  /interview/poll/{pairing_code}    -> Quest polls this every 2s

The pairing store here is in-memory and per-process. That's fine for a single
worker / hackathon demo. For production with multiple workers, swap _PAIRINGS
for a Redis hash with TTL.
"""

import random
import string
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["pairing"])

# ----------------------------------------------------------------------------
# In-memory pairing store
# ----------------------------------------------------------------------------

PAIRING_TTL = timedelta(minutes=15)   # pairing codes expire after 15 min
PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I, O, 0, 1 — avoid confusion
PAIRING_LENGTH = 4

_PAIRINGS: dict[str, dict] = {}   # pairing_code -> {access_code, created_at}
_LOCK = Lock()


def _gen_pairing_code() -> str:
    """Generate a 4-char pairing code, guaranteed not currently in use."""
    while True:
        code = "".join(random.choices(PAIRING_ALPHABET, k=PAIRING_LENGTH))
        if code not in _PAIRINGS:
            return code


def _purge_expired() -> None:
    """Drop expired entries. Cheap, called on every request."""
    now = datetime.now(timezone.utc)
    expired = [c for c, e in _PAIRINGS.items() if now - e["created_at"] > PAIRING_TTL]
    for c in expired:
        _PAIRINGS.pop(c, None)


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------

class NewPairingResponse(BaseModel):
    pairing_code: str
    expires_in_seconds: int


class PairRequest(BaseModel):
    pairing_code: str = Field(..., min_length=PAIRING_LENGTH, max_length=PAIRING_LENGTH)
    access_code: str


class PairResponse(BaseModel):
    status: str


class PollResponse(BaseModel):
    access_code: str | None


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@router.post("/new-pairing", response_model=NewPairingResponse)
def new_pairing():
    """Quest calls this once when the keypad scene loads."""
    with _LOCK:
        _purge_expired()
        code = _gen_pairing_code()
        _PAIRINGS[code] = {
            "access_code": None,
            "created_at": datetime.now(timezone.utc),
        }
    return NewPairingResponse(
        pairing_code=code,
        expires_in_seconds=int(PAIRING_TTL.total_seconds()),
    )


@router.post("/pair", response_model=PairResponse)
def pair(req: PairRequest):
    """Frontend calls this after /setup-interview returns an access_code."""
    code = req.pairing_code.upper().strip()
    with _LOCK:
        _purge_expired()
        entry = _PAIRINGS.get(code)
        if entry is None:
            raise HTTPException(404, "Pairing code not found or expired")
        entry["access_code"] = req.access_code
    return PairResponse(status="paired")


@router.get("/poll/{pairing_code}", response_model=PollResponse)
def poll(pairing_code: str):
    """
    Quest polls this every ~2s after showing the pairing code.

    Returns access_code:null while waiting, then once with the real access_code.
    The pairing entry is deleted on the first successful poll (one-shot) so a
    stale Quest can't accidentally pick up someone else's session later.
    """
    code = pairing_code.upper().strip()
    with _LOCK:
        _purge_expired()
        entry = _PAIRINGS.get(code)
        if entry is None:
            raise HTTPException(404, "Pairing code not found or expired")
        access_code = entry["access_code"]
        if access_code is not None:
            _PAIRINGS.pop(code, None)   # one-shot delivery
    return PollResponse(access_code=access_code)
