import os
import cv2
import numpy as np
import time

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "camera_calibration_params.npz")
WINDOW = "LOGIN + GAME (q/ESC quit | b=select ball | h=add hole | r=reset score)"
# (ancho, alto) de las fotos de calibración, por si el .npz no lo trae guardado
DEFAULT_CALIB_SIZE = (1920, 1080)


def load_calibration(path):
    try:
        data = np.load(path)
        if "image_size" in data:
            calib_size = tuple(int(v) for v in data["image_size"])
        else:
            calib_size = DEFAULT_CALIB_SIZE
        return data["intrinsics"], data["dist_coeffs"], calib_size
    except Exception as e:
        print(f"AVISO: no se pudo cargar la calibración ({path}): {e}")
        print("Se continúa sin corregir la distorsión.")
        return None, None, None


def scale_intrinsics(K, calib_size, frame_size):
    # K está en píxeles de la resolución de calibración: se reescala a la del frame
    sx = frame_size[0] / calib_size[0]
    sy = frame_size[1] / calib_size[1]
    if abs(sx - sy) > 0.01:
        print(f"AVISO: la proporción del frame {frame_size} no coincide con la de "
              f"calibración {calib_size}; la corrección de distorsión puede ser inexacta.")
    K_scaled = K.astype(np.float64).copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def make_undistort_maps(K, dist, calib_size, frame_size):
    if K is None or dist is None:
        return None
    K_frame = scale_intrinsics(K, calib_size, frame_size)
    return cv2.initUndistortRectifyMap(K_frame, dist, None, K_frame,
                                       frame_size, cv2.CV_16SC2)


def maybe_undistort(frame, maps):
    if maps is None:
        return frame
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)


def shape_from_contour(cnt):
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return None

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx)

    area = cv2.contourArea(cnt)
    if area <= 1e-6:
        return None

    circ = 4 * np.pi * area / (peri * peri)

    if v == 3:
        return "TRIANGLE"

    if v == 4:
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w <= 1e-6 or h <= 1e-6:
            return None
        ar = max(w, h) / min(w, h)
        if ar > 1.3:
            return None
        angle = rect[2]
        ang = abs(angle)
        if 20 < ang < 70:
            return "DIAMOND"
        return "SQUARE"

    if v >= 8 and circ > 0.80:
        return "CIRCLE"

    return None


def detect_tokens_in_roi(roi_bgr, min_area=600):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    color_ranges = {
        "RED": [
            (np.array([0, 90, 70]), np.array([10, 255, 255])),
            (np.array([170, 90, 70]), np.array([179, 255, 255]))
        ],
        "GREEN": [
            (np.array([40, 70, 70]), np.array([85, 255, 255]))
        ],
        "YELLOW": [
            (np.array([20, 80, 80]), np.array([35, 255, 255]))
        ],
        "BLUE": [
            (np.array([90, 80, 60]), np.array([130, 255, 255]))
        ]
    }

    tokens = set()
    boxes = []

    for color_name, ranges in color_ranges.items():
        mask = None
        for (lo, hi) in ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            shape = shape_from_contour(c)
            if shape is None:
                continue

            x, y, w, h = cv2.boundingRect(c)
            tokens.add((color_name, shape))
            boxes.append((x, y, w, h, color_name, shape))

    return tokens, boxes


def draw_token_boxes(roi, display, boxes, offset_x, offset_y):
    color_bgr = {
        "RED": (0, 0, 255),
        "GREEN": (0, 255, 0),
        "YELLOW": (0, 255, 255),
        "BLUE": (255, 0, 0)
    }

    for (x, y, w, h, cname, sname) in boxes:
        bgr = color_bgr.get(cname, (255, 255, 255))
        cv2.rectangle(roi, (x, y), (x + w, y + h), bgr, 2)
        cv2.rectangle(display, (offset_x + x, offset_y + y),
                      (offset_x + x + w, offset_y + y + h), bgr, 2)
        cv2.putText(display, f"{cname}-{sname}",
                    (offset_x + x, offset_y + y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)


LOW_BLUE = np.array([95, 80, 60])
HIGH_BLUE = np.array([130, 255, 255])
KERNEL_BLUE = np.ones((5, 5), np.uint8)

MIN_AREA = 250
SEARCH_MARGIN = 200
MAX_JUMP_PX = 120
LOST_LIMIT = 25
INSIDE_CONFIRM_FRAMES = 8


def crop_roi(img, center, margin):
    H, W = img.shape[:2]
    cx, cy = center
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(W, cx + margin)
    y2 = min(H, cy + margin)
    return img[y1:y2, x1:x2], x1, y1


def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)


def detect_blue_balls(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW_BLUE, HIGH_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_BLUE, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_BLUE, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 2:
            continue
        circles.append((int(x), int(y), int(r)))

    return circles


def detect_blue_balls_near(frame_bgr, last_center, margin):
    roi, ox, oy = crop_roi(frame_bgr, last_center, margin)
    circles_roi = detect_blue_balls(roi)
    return [(cx + ox, cy + oy, r) for (cx, cy, r) in circles_roi]


def main(camera_index=0, width=1280, height=720):
    K, dist, calib_size = load_calibration(CALIB_PATH)
    undistort_maps = None
    maps_size = None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    password = [
        ("RED", "TRIANGLE"),
        ("GREEN", "SQUARE"),
        ("RED", "CIRCLE"),
        ("BLUE", "DIAMOND")
    ]

    mode = "LOGIN"
    step = 0
    expected_was_present = False

    selecting_ball = False
    click_point = None
    tracked_circle = None
    lost_frames = 0
    holes = []
    score = 0
    inside_history = []
    already_scored_this_loss = False

    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def on_mouse(event, x, y, flags, param):
        nonlocal selecting_ball, click_point
        if event == cv2.EVENT_LBUTTONDOWN and selecting_ball:
            click_point = (x, y)
            selecting_ball = False

    cv2.setMouseCallback(WINDOW, on_mouse)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # la cámara puede no respetar width/height: se usa el tamaño real del frame
        frame_size = (frame.shape[1], frame.shape[0])
        if frame_size != maps_size:
            undistort_maps = make_undistort_maps(K, dist, calib_size, frame_size)
            maps_size = frame_size
        frame = maybe_undistort(frame, undistort_maps)
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        H, W = frame.shape[:2]

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 1.0 / dt

        if mode == "LOGIN":
            roi_size = int(min(H, W) * 0.38)
            x1, y1 = 20, 20
            roi_size = min(roi_size, W - x1 - 1, H - y1 - 1)
            x2, y2 = x1 + roi_size, y1 + roi_size
            roi = frame[y1:y2, x1:x2]

            tokens, boxes = detect_tokens_in_roi(roi, min_area=600)
            draw_token_boxes(roi, display, boxes, x1, y1)

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(display, "FIGURAS",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            expected = password[step]
            expected_present = expected in tokens

            if expected_present and not expected_was_present:
                step += 1
                expected_was_present = True
                if step >= len(password):
                    mode = "GAME"
                    selecting_ball = True
                    click_point = None
                    tracked_circle = None
                    lost_frames = 0
                    inside_history.clear()
                    already_scored_this_loss = False
            elif not expected_present:
                expected_was_present = False

            cv2.putText(display, f"{step}/{len(password)}",
                        (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        else:
            if click_point is not None and tracked_circle is None:
                circles = detect_blue_balls(frame)
                if circles:
                    cx0, cy0 = click_point
                    tracked_circle = min(circles, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)
                    lost_frames = 0
                    inside_history.clear()
                    already_scored_this_loss = False
                click_point = None

            if tracked_circle is not None:
                cx, cy, r = tracked_circle
                circles = detect_blue_balls_near(frame, (cx, cy), SEARCH_MARGIN)

                if circles:
                    best = min(circles, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
                    if dist2((best[0], best[1]), (cx, cy)) <= MAX_JUMP_PX * MAX_JUMP_PX:
                        tracked_circle = best
                        lost_frames = 0
                    else:
                        lost_frames += 1
                else:
                    lost_frames += 1

                inside_now = any(point_in_rect(tracked_circle[0], tracked_circle[1], hrect) for hrect in holes) if holes else False
                inside_history.append(inside_now)
                if len(inside_history) > INSIDE_CONFIRM_FRAMES:
                    inside_history.pop(0)

                if not inside_now:
                    already_scored_this_loss = False

                if lost_frames > LOST_LIMIT:
                    inside_count = sum(1 for v in inside_history if v)
                    was_inside_recently = inside_count >= (INSIDE_CONFIRM_FRAMES // 2 + 1)
                    if was_inside_recently and not already_scored_this_loss:
                        score += 1
                        already_scored_this_loss = True
                    tracked_circle = None
                    lost_frames = 0
                    inside_history.clear()

            for i, (x, y, w, h) in enumerate(holes):
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display, f"H{i + 1}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if tracked_circle is not None:
                cx, cy, r = tracked_circle
                x, y, w, h = cx - r, cy - r, 2 * r, 2 * r
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

                if holes and any(point_in_rect(cx, cy, hrect) for hrect in holes):
                    cv2.putText(display, "IN HOLE ZONE", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.putText(display, f"lost:{lost_frames}", (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display, f"Score: {score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if selecting_ball:
                cv2.putText(display, "Click en la bola azul...",
                            (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display, "b=select ball | h=add hole ROI | r=reset | q=quit",
                            (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        cv2.putText(display, f"FPS: {fps:.1f}",
                    (W - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        if mode == "GAME":
            if key == ord('b'):
                selecting_ball = True
                click_point = None
                tracked_circle = None
                lost_frames = 0
                inside_history.clear()
                already_scored_this_loss = False

            if key == ord('h'):
                roi = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
                cv2.setMouseCallback(WINDOW, on_mouse)
                if roi[2] > 0 and roi[3] > 0:
                    holes.append(tuple(map(int, roi)))

            if key == ord('r'):
                score = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
