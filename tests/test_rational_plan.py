from __future__ import annotations

import unittest
from fractions import Fraction

from timeline_engine.errors import ProjectValidationError
from timeline_engine.model import OutputSpec, Project, validate_project
from timeline_engine.rational import (
    RationalError,
    ceil_div,
    round_half_up,
    to_fraction,
)
from timeline_engine.timeline import build_plan, plan_segments


OUTPUT = OutputSpec(64, 64, 30000, 1001, 48000, 2)


class RationalTests(unittest.TestCase):
    def test_float_is_rejected(self) -> None:
        with self.assertRaises(RationalError):
            to_fraction(0.1)

    def test_forms(self) -> None:
        self.assertEqual(to_fraction("3/2"), Fraction(3, 2))
        self.assertEqual(to_fraction({"num": -1, "den": 3}), Fraction(-1, 3))
        self.assertEqual(to_fraction(7), Fraction(7))

    def test_round_half_up(self) -> None:
        self.assertEqual(round_half_up(Fraction(5, 2)), 3)
        self.assertEqual(round_half_up(Fraction(-5, 2)), -3)
        self.assertEqual(ceil_div(7, 3), 3)


class PlanTests(unittest.TestCase):
    def test_exact_quantization(self) -> None:
        project = Project(OUTPUT, Fraction(1, 3), {}, [], [], [])
        plan = build_plan(project)
        self.assertEqual(plan.frame_count, 10)
        self.assertEqual(plan.normalized_duration, Fraction(1001, 3000))
        self.assertEqual(plan.sample_count, 16016)

    def test_segment_partition_is_disjoint_and_complete(self) -> None:
        project = Project(OUTPUT, Fraction(37, 7), {}, [], [], [])
        plan = build_plan(project)
        for size in (1, 7, 45, 1000):
            segments = plan_segments(plan, size)
            self.assertEqual(
                sum(item.frame_count for item in segments), plan.frame_count
            )
            self.assertEqual(
                sum(item.sample_count for item in segments), plan.sample_count
            )
            for left, right in zip(segments, segments[1:]):
                self.assertEqual(left.frame_end, right.frame_start)
                self.assertEqual(left.sample_end, right.sample_start)

    def test_invalid_duration(self) -> None:
        project = Project(OUTPUT, Fraction(0), {}, [], [], [])
        with self.assertRaises(ProjectValidationError):
            validate_project(project)


if __name__ == "__main__":
    unittest.main()
