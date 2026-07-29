from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import load_config
from .maix_app import draw_solution
from .pipeline import rectify_frame, solve_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve one calibrated E-topic puzzle image.")
    parser.add_argument("--image", required=True, type=Path, help="BGR image captured by the puzzle camera")
    parser.add_argument("--config", required=True, type=Path, help="calibrated JSON configuration")
    parser.add_argument("--output", type=Path, help="optional annotated BGR image path")
    args = parser.parse_args(argv)
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error(f"cannot read image: {args.image}")
    config = load_config(args.config)
    result = solve_frame(frame, config)
    if args.output is not None:
        board = rectify_frame(frame, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), draw_solution(board if board is not None else frame, result, config)):
            parser.error(f"cannot write output: {args.output}")
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
