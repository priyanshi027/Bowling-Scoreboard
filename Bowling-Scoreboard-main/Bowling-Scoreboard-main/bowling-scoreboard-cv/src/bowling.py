"""Stage 6 - bowling scoring rules, used to validate and repair the reading.

This is the accuracy lever that a pure OCR pipeline does not have. The board
prints both the throws and the running total, and the two are linked by the
rules of ten-pin bowling. Recomputing the totals from the throws and comparing
against the printed totals turns "here is what the pixels looked like" into a
result that can be checked - and, where the throws agree but a printed total
does not, the computed value can be preferred.

Notation on the board:
    X       strike
    /       spare (the frame's two throws add to ten)
    -       a miss, zero pins
    digit   that many pins
    Ndigit  followed by lowercase "o" - a circled digit, the board's split
            marker. It annotates the throw; it does not change the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

N_FRAMES = 10
UNKNOWN = "?"


# ------------------------------------------------------------------ tokenising


def tokenize(text: str) -> list[str]:
    """Split a cell reading into glyph tokens.

    A lowercase ``o`` is only ever emitted as a circled-digit suffix, so it can
    be folded into the preceding token without ambiguity.
    """
    tokens: list[str] = []
    for ch in text.strip():
        if ch == "o" and tokens and tokens[-1].isdigit():
            tokens[-1] += "o"
        else:
            tokens.append(ch)
    return tokens


def pins(token: str) -> int | None:
    """Pin count for a single throw token, or ``None`` if unreadable."""
    base = token[:-1] if token.endswith("o") else token
    if base == "X":
        return 10
    if base == "-":
        return 0
    if base.isdigit():
        return int(base)
    return None                     # "/" is contextual; "?" is a misread


def parse_int(text: str) -> int | None:
    """Read a running-total cell. Returns ``None`` if blank or unreadable."""
    text = text.strip()
    if not text or not text.isdigit():
        return None
    return int(text)


# --------------------------------------------------------------------- frames


@dataclass
class Frame:
    index: int                      # 1-based
    raw: str = ""                   # text as read off the board
    rolls: list[int] = field(default_factory=list)
    split: bool = False
    complete: bool = False
    readable: bool = True
    frame_score: int | None = None  # points credited to this frame, with bonuses
    cumulative: int | None = None   # computed running total
    printed: int | None = None      # running total as printed on the board

    @property
    def played(self) -> bool:
        return bool(self.rolls)


def parse_frame(text: str, index: int) -> Frame:
    """Turn a throws-cell reading into pin counts."""
    frame = Frame(index=index, raw=text)
    tokens = tokenize(text)
    if not tokens:
        return frame

    frame.split = any(t.endswith("o") for t in tokens)
    if UNKNOWN in tokens:
        frame.readable = False
        return frame

    rolls: list[int] = []
    standing = 10
    for token in tokens:
        if token == "/":
            # A spare mark stands for "whatever was left after the last throw".
            if not rolls:
                frame.readable = False
                return frame
            rolls.append(standing)
            standing = 10
            continue

        value = pins(token)
        if value is None:
            frame.readable = False
            return frame
        rolls.append(value)

        if value == 10 and standing == 10:
            standing = 10           # strike: a fresh rack follows
        else:
            standing -= value
            if standing < 0:
                frame.readable = False
                return frame
            if standing == 0:
                standing = 10       # rack cleared, pins reset

    frame.rolls = rolls
    frame.complete = _is_complete(rolls, index)
    return frame


def _is_complete(rolls: list[int], index: int) -> bool:
    if not rolls:
        return False
    if index < N_FRAMES:
        return rolls[0] == 10 or len(rolls) >= 2
    # The tenth frame earns a third ball after a strike or a spare.
    if rolls[0] == 10 or (len(rolls) >= 2 and rolls[0] + rolls[1] == 10):
        return len(rolls) >= 3
    return len(rolls) >= 2


# -------------------------------------------------------------------- scoring


def score_frames(frames: list[Frame]) -> None:
    """Fill in ``frame_score`` and ``cumulative`` in place.

    A frame whose bonus balls have not been thrown yet stays ``None``, and so
    does every frame after it - a running total is only meaningful once every
    earlier frame has settled.
    """
    flat: list[int] = []
    starts: list[int] = []
    for frame in frames:
        starts.append(len(flat))
        flat.extend(frame.rolls)

    running = 0
    settled = True
    for i, frame in enumerate(frames):
        frame.frame_score = None
        frame.cumulative = None
        if not settled or not frame.played or not frame.readable:
            settled = False
            continue

        start = starts[i]
        if frame.index < N_FRAMES:
            if frame.rolls[0] == 10:                              # strike
                bonus = flat[start + 1:start + 3]
                if len(bonus) < 2:
                    settled = False
                    continue
                points = 10 + sum(bonus)
            elif len(frame.rolls) >= 2 and frame.rolls[0] + frame.rolls[1] == 10:
                bonus = flat[start + 2:start + 3]                 # spare
                if len(bonus) < 1:
                    settled = False
                    continue
                points = 10 + bonus[0]
            elif len(frame.rolls) >= 2:                            # open frame
                points = frame.rolls[0] + frame.rolls[1]
            else:
                settled = False                                    # mid-frame
                continue
        else:
            if not frame.complete:
                settled = False
                continue
            points = sum(frame.rolls)

        running += points
        frame.frame_score = points
        frame.cumulative = running


# ------------------------------------------------------------------ validation


@dataclass
class PlayerScore:
    initial: str
    frames: list[Frame]
    printed_total: int | None = None
    computed_total: int | None = None
    mismatched_frames: list[int] = field(default_factory=list)
    repaired_frames: list[int] = field(default_factory=list)

    @property
    def checked(self) -> int:
        """Frames where a printed total and a computed total both exist."""
        return sum(
            1 for f in self.frames if f.printed is not None and f.cumulative is not None
        )

    @property
    def agreement(self) -> float:
        checked = self.checked
        if not checked:
            return 1.0
        return (checked - len(self.mismatched_frames)) / checked

    @property
    def total(self) -> int | None:
        """Running total over the frames that have fully settled."""
        for frame in reversed(self.frames):
            if frame.cumulative is not None:
                return frame.cumulative
        return None

    @property
    def provisional_total(self) -> int | None:
        """Settled total plus pins already down in an open in-progress frame.

        This is what the board's TTL column shows. Pins from an unresolved strike
        or spare are *not* added, because their bonus is still unknown - matching
        the board, which holds the total back until the bonus balls are thrown.
        """
        settled = self.total
        base = settled if settled is not None else 0

        for frame in self.frames:
            if frame.cumulative is not None or not frame.played:
                continue
            if not frame.readable:
                break
            # First unsettled frame: credit it only if it is still open.
            is_strike = frame.rolls[0] == 10
            is_spare = len(frame.rolls) >= 2 and frame.rolls[0] + frame.rolls[1] == 10
            if not is_strike and not is_spare:
                base += sum(frame.rolls)
            break

        if settled is None and base == 0:
            return None
        return base

    @property
    def total_agrees(self) -> bool | None:
        """Whether the printed TTL matches the provisional total."""
        if self.printed_total is None or self.provisional_total is None:
            return None
        return self.printed_total == self.provisional_total


def build_player(initial: str, throw_texts: list[str], score_texts: list[str],
                 total_text: str, repair: bool = True) -> PlayerScore:
    """Assemble one player's row into a validated result.

    ``throw_texts`` and ``score_texts`` are the ten throw cells and the ten
    running-total cells, in frame order.
    """
    frames = [parse_frame(throw_texts[i], i + 1) for i in range(N_FRAMES)]
    for i, frame in enumerate(frames):
        frame.printed = parse_int(score_texts[i])

    score_frames(frames)

    player = PlayerScore(initial=initial, frames=frames)
    player.printed_total = parse_int(total_text)

    for frame in frames:
        if frame.printed is None or frame.cumulative is None:
            continue
        if frame.printed != frame.cumulative:
            player.mismatched_frames.append(frame.index)
            if repair:
                # The throws are the primary evidence and the rules are exact, so
                # a disagreement points at the printed digits being misread.
                frame.printed = frame.cumulative
                player.repaired_frames.append(frame.index)

    player.computed_total = player.provisional_total
    return player


def perfect_game() -> list[str]:
    """Twelve strikes - the 300 game. Used by the tests."""
    return ["X"] * 9 + ["XXX"]
