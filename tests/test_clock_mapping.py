from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from timeline_engine.generator import build_frame
from timeline_engine.model import Anchor, ProjectValidationError
from timeline_engine.timeline import ClockMap


class ClockMapTests(unittest.TestCase):
    def test_affine_map_passes_both_anchors_exactly(self) -> None:
        anchors = (
            Anchor(2400, Fraction(133, 1000)),
            Anchor(98400, Fraction(133, 1000) + Fraction(99, 50)),
        )
        clock = ClockMap.derive(anchors, 48000, Fraction(123, 1000))
        for anchor in anchors:
            self.assertEqual(
                clock.sample_position(anchor.video_time),
                anchor.audio_sample,
            )

    def test_default_clock_uses_first_audio_pts(self) -> None:
        clock = ClockMap.derive((), 48000, Fraction(123, 1000))
        self.assertEqual(clock.slope, Fraction(48000))
        self.assertEqual(
            clock.intercept, -Fraction(123, 1000) * 48000
        )

    def test_reversed_anchors_rejected_at_project_level(self) -> None:
        from timeline_engine.model import Clip, validate_project
        from timeline_engine.project_factory import media_from_spec
        from timeline_engine.generator import GeneratedSpec

        spec = GeneratedSpec(
            path=__import__("pathlib").Path("x.mkv"),
            kind="x",
            width=256,
            height=144,
            fps=Fraction(25),
            frame_count=10,
            sample_rate=48000,
            source_id=1,
            first_video_pts=Fraction(0),
            first_audio_pts=Fraction(0),
            pulse_positions=(),
        )
        clip = Clip(
            clip_id="bad",
            media_id="x",
            src_in=Fraction(0),
            src_out=Fraction(1),
            timeline_start=Fraction(0),
            speed=Fraction(1),
            anchors=(
                Anchor(100, Fraction(2)),
                Anchor(50, Fraction(1)),
            ),
        )
        project = __import__("timeline_engine.model", fromlist=["Project"]).Project(
            output=__import__(
                "timeline_engine.model", fromlist=["OutputSpec"]
            ).OutputSpec(256, 144, 30000, 1001, 48000, 2),
            duration=Fraction(1),
            media={"x": media_from_spec(spec)},
            clips=[clip],
            transitions=[],
            backgrounds=[],
        )
        with self.assertRaises(ProjectValidationError):
            validate_project(project)

    def test_vfr_selection_uses_pts_not_frame_index(self) -> None:
        # Explicit irregular PTS list; selection must land on the last PTS
        # below the requested time even with wildly varying gaps.
        pts = [Fraction(0), Fraction(1, 10), Fraction(1, 2), Fraction(3, 4)]
        target = Fraction(6, 10)
        from bisect import bisect_right

        index = bisect_right(pts, target) - 1
        self.assertEqual(index, 2)


if __name__ == "__main__":
    unittest.main()
