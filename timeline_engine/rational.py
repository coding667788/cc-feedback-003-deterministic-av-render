"""Exact rational time arithmetic.

Every time, speed and time base in a project is stored as a ratio of two
integers.  JSON accepts either a ``{"num": n, "den": d}`` object, the string
forms ``"n/d"`` / ``"n"`` or a JSON integer.  JSON floating point numbers are
rejected so that no binary float can become an authoritative time value.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable


class RationalError(ValueError):
    """Raised when a value cannot be interpreted as an exact rational."""


def to_fraction(value: Any, *, field: str = "value") -> Fraction:
    """Convert a project value into an exact :class:`Fraction`.

    Accepted forms::

        3                     -> Fraction(3, 1)
        "3/2"                 -> Fraction(3, 2)
        {"num": 3, "den": 2}  -> Fraction(3, 2)
        {"num": 3}            -> Fraction(3, 1)
        Fraction(...)         -> returned unchanged

    ``float`` is rejected explicitly because it cannot represent values such
    as 1/3 exactly and must never become the authoritative timeline.
    """
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, bool):
        raise RationalError(f"{field}: boolean is not a valid rational")
    elif isinstance(value, int):
        result = Fraction(value, 1)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise RationalError(f"{field}: empty rational string")
        if "/" in text:
            num_text, den_text = text.split("/", 1)
            try:
                num = int(num_text.strip())
                den = int(den_text.strip())
            except ValueError as exc:
                raise RationalError(f"{field}: invalid rational string {value!r}") from exc
            result = Fraction(num, den)
        else:
            try:
                result = Fraction(int(text), 1)
            except ValueError as exc:
                raise RationalError(f"{field}: invalid rational string {value!r}") from exc
    elif isinstance(value, dict):
        if "num" not in value:
            raise RationalError(f"{field}: rational object requires 'num'")
        num_raw = value["num"]
        den_raw = value.get("den", 1)
        if isinstance(num_raw, bool) or isinstance(den_raw, bool):
            raise RationalError(f"{field}: boolean is not a valid rational")
        if not isinstance(num_raw, int) or not isinstance(den_raw, int):
            raise RationalError(
                f"{field}: rational num/den must be integers, got {value!r}"
            )
        result = Fraction(num_raw, den_raw)
    else:
        raise RationalError(
            f"{field}: unsupported rational type {type(value).__name__} ({value!r})"
        )
    if result.denominator <= 0:
        raise RationalError(f"{field}: denominator must be a positive integer")
    return result


def to_json(value: Fraction) -> dict[str, int]:
    """Serialize a fraction as the canonical project JSON form."""
    return {"num": value.numerator, "den": value.denominator}


def common_tick_rate(values: Iterable[Fraction]) -> int:
    """Return a positive integer tick rate divisible by every denominator.

    A time ``t`` is represented on the integer timeline as ``t * rate`` and
    every conversion is therefore exact (no per-segment float accumulation).
    """
    rate = 1
    for value in values:
        rate = math.lcm(rate, value.denominator)
        if rate <= 0:
            raise RationalError("non-positive tick rate")
    return rate


def ceil_div(numerator: int, denominator: int) -> int:
    """Exact ``ceil(numerator / denominator)`` for a positive denominator."""
    if denominator <= 0:
        raise RationalError("denominator must be positive")
    return -((-numerator) // denominator)


def round_half_up_div(numerator: int, denominator: int) -> int:
    """Exact ``round half away from zero`` for a positive denominator."""
    if denominator <= 0:
        raise RationalError("denominator must be positive")
    if numerator >= 0:
        return (2 * numerator + denominator) // (2 * denominator)
    return -((2 * -numerator + denominator) // (2 * denominator))


def round_half_up(value: Fraction) -> int:
    """Round an exact fraction, halves going away from zero."""
    return round_half_up_div(value.numerator, value.denominator)


def frac_str(value: Fraction) -> str:
    """Stable human-readable exact fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
