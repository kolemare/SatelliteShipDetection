#!/usr/bin/env python3
"""
Click to visualize centered tiles on a satellite image.
Draws 112×112 (blue), 224×224 (green), 448×448 (red).

Usage:
    python drawtiles.py --image downloads/raw/your_image.png
"""

import argparse
import cv2

SIZES = [112, 224, 448]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B, G, R


def draw_tiles_on_click(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    display = img.copy()
    win = "Click to draw tiles (press any key to exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def draw(x, y):
        nonlocal display
        display = img.copy()
        for size, color in zip(SIZES, COLORS):
            half = size // 2
            x1, y1 = x - half, y - half
            x2, y2 = x + half, y + half
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                f"{size}x{size}",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            draw(x, y)
            cv2.imshow(win, display)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, display)
    print("🖱️ Click anywhere on the image to draw 112/224/448 rectangles.")
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    draw_tiles_on_click(args.image)


if __name__ == "__main__":
    main()
