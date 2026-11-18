"""Game layer: XP, levels, quests, combos and badges for a SignalForge session.

Pure logic (no Streamlit) so it is unit-testable. The app turns user actions into event
strings ("visit:<page>", "ask", "approve", ...) and calls `award` once per rerun.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

LEVELS = [(0, "Intern"), (120, "Associate PM"), (300, "Product Manager"), (550, "Senior PM"), (850, "Group PM"),
          (1200, "Director of Product"), (1650, "VP Product"), (2200, "Chief Product Officer")]

# id, title, xp, hint, page where it is done (ordered = suggested mission path)
QUESTS = [
    ("tour", "Explore every room", 60, "Open all 7 pages from the navigation.", "Customer Intelligence"),
    ("ask", "Consult SIGNAL", 40, "Ask SIGNAL any question, or tap a quick question.", "Executive Dashboard"),
    ("curate", "Shape a theme", 60, "Rename, merge, split or reject a theme.", "Customer Intelligence"),
    ("constrain", "Bend the roadmap", 60, "Add a dependency or a ranked override.", "Roadmap Optimizer"),
    ("propose", "Propose a decision", 80, "Pick an opportunity and propose it.", "Decision Loop"),
    ("approve", "Get it approved", 100, "Approve the proposed decision.", "Decision Loop"),
    ("ticket", "Hand off to engineering", 100, "Create the ticket and PRD.", "Decision Loop"),
    ("ship", "Ship it", 120, "Record the release.", "Decision Loop"),
    ("measure", "Prove it worked", 150, "Measure the outcome against target.", "Decision Loop"),
    ("debate", "Survive the boardroom", 50, "Ask SIGNAL to run the decision debate.", "Scenario Lab"),
    ("publish", "Brief the board", 80, "Download a memo, PRD or decision pack.", "Decision Memo & PRD"),
]
QUEST_IDS = [q[0] for q in QUESTS]
VISIT_XP = 15
COMBO_WINDOW_S = 120
MAX_COMBO = 3


@dataclass
class Player:
    xp: int = 0
    done: list[str] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)
    combo: int = 1
    last_action: float | None = None
    started: float = field(default_factory=time.time)
    badges: list[str] = field(default_factory=list)


def level(xp: int) -> tuple[int, str, int, int | None]:
    """(level number 1-based, title, xp at this level, xp needed for next level or None at max)."""
    idx = max(i for i, (need, _) in enumerate(LEVELS) if xp >= need)
    nxt = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    return idx + 1, LEVELS[idx][1], LEVELS[idx][0], nxt


def progress(xp: int) -> float:
    _, _, lo, hi = level(xp)
    return 1.0 if hi is None else (xp - lo) / (hi - lo)


def next_quest(p: Player) -> tuple | None:
    return next((q for q in QUESTS if q[0] not in p.done), None)


def award(p: Player, events: set[str], n_pages: int, now: float | None = None) -> list[tuple[str, str]]:
    """Apply events; return notifications as (kind, text) where kind is xp|quest|level|badge."""
    now = time.time() if now is None else now
    out: list[tuple[str, str]] = []
    before = level(p.xp)[0]
    gained = 0
    for ev in sorted(events):
        if ev.startswith("visit:"):
            pg = ev[6:]
            if pg not in p.visited:
                p.visited.append(pg)
                gained += VISIT_XP
            if len(p.visited) >= n_pages:
                ev = "tour"
            else:
                continue
        if ev in QUEST_IDS and ev not in p.done:
            # combo: quests chained within the window multiply XP (x2, x3)
            p.combo = min(p.combo + 1, MAX_COMBO) if p.last_action is not None and now - p.last_action <= COMBO_WINDOW_S else 1
            p.last_action = now
            q = QUESTS[QUEST_IDS.index(ev)]
            xp = q[2] * p.combo
            p.done.append(ev)
            gained += xp
            out.append(("quest", f"Quest complete: {q[1]}  +{xp} XP" + (f"  (combo x{p.combo})" if p.combo > 1 else "")))
    p.xp += gained
    for badge, ok in [("Closed the loop", all(q in p.done for q in ("propose", "approve", "ticket", "ship", "measure"))),
                      ("Speedrunner", "measure" in p.done and now - p.started <= 600),
                      ("Combo master", p.combo >= MAX_COMBO),
                      ("Completionist", len(p.done) == len(QUESTS))]:
        if ok and badge not in p.badges:
            p.badges.append(badge)
            out.append(("badge", f"Badge unlocked: {badge}"))
    after = level(p.xp)
    if after[0] > before:
        out.append(("level", f"Level up! You are now Lv {after[0]} · {after[1]}"))
    return out
