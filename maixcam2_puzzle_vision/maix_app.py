from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, load_config
from .models import PlanResult, SolveStatus
from .pipeline import rectify_frame, solve_frame


LONG_PRESS_MS = 800


def application_paths(module_path: Path | None = None) -> tuple[str, str]:
    package_file = Path(__file__).resolve() if module_path is None else Path(module_path)
    app_directory = package_file.parent.parent
    return (
        (app_directory / "config" / "maixcam2_puzzle_vision.json").as_posix(),
        (app_directory / "calibration.jpg").as_posix(),
    )


def key_release_action(duration_ms: int, long_press_ms: int = LONG_PRESS_MS) -> str:
    return "capture" if duration_ms >= long_press_ms else "solve"


class SolveRequest:
    """A one-bit request consumed exactly once by the camera loop."""

    def __init__(self) -> None:
        self._pending = False

    def request(self) -> None:
        self._pending = True

    def consume(self) -> bool:
        if not self._pending:
            return False
        self._pending = False
        return True


def draw_solution(board_bgr: np.ndarray, result: PlanResult, config: AppConfig) -> np.ndarray:
    overlay = board_bgr.copy()
    scale = config.board.pixels_per_mm
    status_color = (0, 200, 0) if result.status is SolveStatus.OK else (0, 0, 220)
    target_color = (255, 0, 255)
    target_outline = (0, 0, 0)
    cv2.putText(
        overlay,
        f"{result.status.value} confidence={result.confidence:.2f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )
    for command in result.commands:
        pick = tuple(int(round(value * scale)) for value in command.pick_xy_mm)
        place = tuple(int(round(value * scale)) for value in command.place_xy_mm)
        theta = math.radians(command.delta_theta_deg)
        endpoint = (int(round(place[0] + 20 * math.cos(theta))), int(round(place[1] + 20 * math.sin(theta))))
        cv2.circle(overlay, pick, 5, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, place, 8, target_outline, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, place, 6, target_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(overlay, place, endpoint, target_outline, 4, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(overlay, place, endpoint, target_color, 2, cv2.LINE_AA, tipLength=0.25)
        label = f"P{command.piece_id} {command.delta_theta_deg:+.1f}"
        label_origin = (place[0] + 8, place[1] - 8)
        cv2.putText(overlay, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, target_outline, 3, cv2.LINE_AA)
        cv2.putText(overlay, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, target_color, 1, cv2.LINE_AA)
    return overlay


def main(config_path: str | None = None) -> None:
    # Imports stay local so desktop tests can import this module without MaixPy.
    from maix import app, camera, display, image, key, time

    default_config_path, calibration_image_path = application_paths()
    config = load_config(config_path or default_config_path)
    cam = camera.Camera(config.camera.width, config.camera.height, image.Format.FMT_BGR888)
    screen = display.Display()
    solve_request = SolveRequest()
    capture_request = SolveRequest()
    press_started_ms: int | None = None
    latest_overlay: np.ndarray | None = None

    def on_key(_key_id: int, state: int) -> None:
        nonlocal press_started_ms
        now_ms = time.ticks_ms()
        if state == key.State.KEY_PRESSED:
            press_started_ms = now_ms
            return
        if state != key.State.KEY_RELEASED or press_started_ms is None:
            return
        action = key_release_action(max(0, now_ms - press_started_ms))
        press_started_ms = None
        (capture_request if action == "capture" else solve_request).request()

    physical_key = key.Key(on_key)
    _ = physical_key
    while not app.need_exit():
        maix_frame = cam.read()
        frame_bgr = image.image2cv(maix_frame, ensure_bgr=True, copy=True)
        if capture_request.consume():
            maix_frame.save(calibration_image_path)
            latest_overlay = frame_bgr.copy()
            cv2.putText(latest_overlay, "Saved calibration.jpg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2, cv2.LINE_AA)
            print(json.dumps({"status": "CALIBRATION_CAPTURED", "path": calibration_image_path}), flush=True)
        if solve_request.consume():
            result = solve_frame(frame_bgr, config)
            board = rectify_frame(frame_bgr, config)
            latest_overlay = draw_solution(board if board is not None else frame_bgr, result, config)
            print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)
        screen.show(image.cv2image(frame_bgr if latest_overlay is None else latest_overlay, bgr=True, copy=False))


if __name__ == "__main__":
    main()
