"""Command line entry points for generation, rendering and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generator import generate_suite
from .model import load_project
from .project_factory import (
    build_anchor_project,
    build_demo_project,
    build_invalid_project,
)
from .render import RenderConfig, render_project
from .source import SourceStore
from .validator import (
    OutputValidator,
    compare_render_files_streaming,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_generate(args: argparse.Namespace) -> int:
    specs = generate_suite(args.output_dir, width=args.width, height=args.height)
    demo = build_demo_project(specs, width=args.width, height=args.height)
    demo.write_json(Path(args.output_dir) / "demo_project.json")
    invalid = build_invalid_project(specs)
    invalid.write_json(Path(args.output_dir) / "invalid_project.json")
    anchor = build_anchor_project(specs)
    anchor.write_json(Path(args.output_dir) / "anchor_project.json")
    print(json.dumps({"generated": [spec.kind for spec in specs]}, indent=2))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    config = RenderConfig(
        job_dir=Path(args.job_dir),
        segment_frames=args.segment_frames,
        parallelism=args.parallelism,
        make_mp4=not args.no_mp4,
    )
    result = render_project(project, config)
    _write_json(Path(args.job_dir) / "render_report.json", result)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    toolchain = json.loads(
        (Path(args.job_dir) / "render_report.json").read_text(encoding="utf-8")
    )["toolchain"]
    store = SourceStore(
        Path(args.job_dir) / "source_cache", project.output, toolchain
    )
    validator = OutputValidator(
        project, store, Path(args.job_dir) / "validation_work"
    )
    report = validator.validate_lossless(args.master_mkv)
    payload = report.to_dict()
    _write_json(Path(args.job_dir) / "validation_report.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if report.ok else 2


def cmd_compare(args: argparse.Namespace) -> int:
    project = load_project(args.project)
    result = compare_render_files_streaming(
        project,
        args.baseline_mkv,
        args.candidate_mkv,
        Path(args.work_dir),
        audio_tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timeline-engine",
        description="Deterministic multi-timebase A/V timeline renderer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic source suite")
    gen.add_argument("output_dir")
    gen.add_argument("--width", type=int, default=256)
    gen.add_argument("--height", type=int, default=144)
    gen.set_defaults(func=cmd_generate)

    render = sub.add_parser("render", help="render a project")
    render.add_argument("project")
    render.add_argument("job_dir")
    render.add_argument("--segment-frames", type=int, default=90)
    render.add_argument("--parallelism", type=int, default=1)
    render.add_argument("--no-mp4", action="store_true")
    render.set_defaults(func=cmd_render)

    validate = sub.add_parser("validate", help="independently validate a master")
    validate.add_argument("project")
    validate.add_argument("job_dir")
    validate.add_argument("master_mkv")
    validate.set_defaults(func=cmd_validate)

    compare = sub.add_parser("compare", help="compare two lossless masters")
    compare.add_argument("project")
    compare.add_argument("baseline_mkv")
    compare.add_argument("candidate_mkv")
    compare.add_argument("--work-dir", default=".compare_work")
    compare.add_argument("--tolerance", type=float, default=1e-6)
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
