"""
api_server.py — Auricular Ear Seed Protocol · standalone server
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date
import anthropic
import os
import logging

from bazi_calculator import (
    get_four_pillars,
    get_element_counts,
    interpret_constitution,
    spread_score,
    is_balanced,
    STATE_RANK,
)
from prompt_builder import SYSTEM_PROMPT, build_user_message
from treatment_protocol import get_protocol, AURICULAR_POINTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ear Seed Protocol", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    name:       Optional[str] = Field(default="")
    year:       int           = Field(..., ge=1900, le=2100)
    month:      int           = Field(..., ge=1,    le=12)
    day:        int           = Field(..., ge=1,    le=31)
    hour:       Optional[int] = Field(default=None)
    handedness: str           = Field(default="right")

    @validator("year")
    def not_future(cls, v):
        if v > date.today().year:
            raise ValueError("Birth year cannot be in the future.")
        return v


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main endpoint ──────────────────────────────────────────────────────────────

@app.post("/generate")
def generate(data: GenerateRequest):
    """
    Single endpoint: Ba Zi → Five Elements → Claude reading + ear seed protocol.
    No email, no logging, no external services beyond the Anthropic API.
    """
    hour_known = data.hour is not None
    calc_hour  = data.hour if hour_known else 12

    # 1. Four Pillars
    try:
        pillars = get_four_pillars(data.year, data.month, data.day, calc_hour)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Date error: {e}")

    if not hour_known:
        pillars.pop("Hour", None)

    # 2. Five Element constitution
    counts       = get_element_counts(pillars)
    constitution = interpret_constitution(counts)
    spread       = spread_score(constitution)
    balanced     = is_balanced(constitution)
    sorted_elems = sorted(constitution.items(), key=lambda x: STATE_RANK[x[1]])
    weakest      = sorted_elems[0][0]
    strongest    = sorted_elems[-1][0]

    # 3. Ear seed protocol
    handedness = "left" if str(data.handedness).lower().startswith("l") else "right"
    try:
        principle, protocol = get_protocol(pillars, constitution, handedness)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Protocol error: {e}")

    # 4. Claude reading
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")

    user_message = build_user_message(
        name        = data.name or "Friend",
        pillars     = pillars,
        constitution= constitution,
        spread      = spread,
        is_balanced = balanced,
        weakest     = weakest,
        strongest   = strongest,
        hour_known  = hour_known,
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        msg = client.messages.create(
            model      = "claude-opus-4-6",
            max_tokens = 1600,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )
        reading_text = msg.content[0].text
        logger.info("Reading generated for %s", data.name or "unnamed")
    except Exception as e:
        logger.error("Claude error: %s", e)
        raise HTTPException(status_code=502, detail=f"Claude error: {e}")

    # 5. Enrich protocol points with body-acupuncture correspondences
    points_out = []
    for p in protocol.points:
        db = AURICULAR_POINTS.get(p.name, {})
        points_out.append({
            "name":              p.name,
            "ear":               p.ear,
            "metal":             p.metal,
            "intent":            p.intent,
            "action":            p.action,
            "point_type":        p.point_type,
            "note":              p.note,
            "body_point_tonify": db.get("body_point_tonify", ""),
            "body_point_sedate": db.get("body_point_sedate", ""),
        })

    return {
        "name":         data.name or "",
        "pillars":      {k: list(v) for k, v in pillars.items()},
        "constitution": constitution,
        "reading":      reading_text,
        "principle":    principle.principle,
        "description":  principle.description,
        "day_master":   principle.day_master,
        "deficient":    principle.deficient,
        "excess":       principle.excess,
        "protocol": {
            "points":     points_out,
            "left_ear":   protocol.left_ear,
            "right_ear":  protocol.right_ear,
            "bilateral":  protocol.bilateral,
            "handedness": protocol.handedness,
        },
    }


# ── Static files — must be last so API routes take priority ───────────────────
app.mount("/", StaticFiles(directory=".", html=True), name="static")
