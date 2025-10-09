#!/usr/bin/env python3
"""
Show an image where the mouse cursor is visualized as 4 centered squares:
80, 112, 224, and 448 px. No output is saved.

Usage:
  python cursor_squares.py path/to/image.jpg

Controls:
  - m : toggle mouse-follow (freeze/unfreeze at current position)
  - c : toggle crosshair
  - t : cycle thickness (1, 2, 3, 4)
  - q / Esc : quit
"""

import sys
from pathlib import Path
import cv2

# Fixed square side lengths (in pixels)
SIZES = (80, 112, 224, 448)

# Colors (BGR): yellow, red, green, blue
COLORS = ((0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0))

STATE = {
    "pos": None,         # (x, y)
    "follow": True,      # if True, squares follow the mouse
    "show_cross": True,  # show crosshair
    "thickness": 2,      # rectangle line thickness
}

def on_mouse(event, x, y, flags, param):
    if STATE["follow"] and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
        STATE["pos"] = (x, y)

def draw_overlay(base_img):
    """Return a copy of base_img with squares (and optional crosshair) drawn."""
    vis = base_img.copy()
    if STATE["pos"] is None:
        return vis

    x, y = STATE["pos"]
    th = STATE["thickness"]

    # draw squares centered at (x, y)
    for size, color in zip(SIZES, COLORS):
        half = size // 2
        x1, y1 = x - half, y - half
        x2, y2 = x + half, y + half
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, th)
        cv2.putText(
            vis, f"{size}x{size}", (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )

    if STATE["show_cross"]:
        # crosshair lines
        h, w = vis.shape[:2]
        cv2.line(vis, (0, y), (w, y), (200, 200, 200), 1)
        cv2.line(vis, (x, 0), (x, h), (200, 200, 200), 1)
        # small center dot
        cv2.circle(vis, (x, y), 3, (220, 220, 220), -1)

    # HUD text
    hud = "m:toggle follow | c:crosshair | t:thickness | q/ESC:quit"
    cv2.rectangle(vis, (8, 8), (8 + 8 * len(hud), 34), (0, 0, 0), -1)
    cv2.putText(vis, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return vis

def main():
    if len(sys.argv) < 2:
        print("Usage: python cursor_squares.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ could not read image: {img_path}")
        sys.exit(1)

    win = "Cursor Squares"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    # initialize position at image center
    h, w = img.shape[:2]
    STATE["pos"] = (w // 2, h // 2)

    while True:
        vis = draw_overlay(img)
        cv2.imshow(win, vis)
        key = cv2.waitKey(16) & 0xFF

        if key in (27, ord('q')):   # ESC / q
            break
        elif key == ord('m'):       # toggle follow
            STATE["follow"] = not STATE["follow"]
        elif key == ord('c'):       # toggle crosshair
            STATE["show_cross"] = not STATE["show_cross"]
        elif key == ord('t'):       # cycle thickness
            STATE["thickness"] = 1 + (STATE["thickness"] % 4)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
