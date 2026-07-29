from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from maixcam2_puzzle_vision.calibration import update_homography_from_corners


CORNER_NAMES = ("TL", "TR", "BR", "BL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Click four A4 corners and update a puzzle vision config.")
    parser.add_argument("--image", required=True, type=Path, help="calibration photo from the current camera position")
    parser.add_argument("--config", required=True, type=Path, help="configuration file to update")
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    points: list[tuple[float, float]] = []
    window = "A4 calibration: click TL TR BR BL, Enter=save, R=reset, Esc=cancel"

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = image.copy()
        for index, point in enumerate(points):
            location = (int(point[0]), int(point[1]))
            cv2.circle(preview, location, 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(preview, CORNER_NAMES[index], (location[0] + 8, location[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return 1
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (10, 13, 32) and len(points) == 4:
            break
    cv2.destroyAllWindows()
    update_homography_from_corners(args.config, tuple(points))
    print(json.dumps({"status": "CALIBRATED", "config": str(args.config), "image_points": points}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
