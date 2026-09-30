import os
import imageio.v2 as imageio
import numpy as np
from typing import List
import cv2


# Rutas relativas a este script, para que funcione se lance desde donde se lance
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
CORNERS_DIR = os.path.join(BASE_DIR, "esquinas")
OUTPUT_CALIB = os.path.join(DATA_DIR, "camera_calibration_params.npz")

CHESSBOARD_SHAPE = (7, 7)  # esquinas interiores
SQUARE_SIZE = 20           # lado de cada casilla (mm)
N_IMAGES = 9


def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_image(output_folder: str, img_name: str, img: np.array):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):
    points = []
    for j in range(chessboard_shape[0]):
        for i in range(chessboard_shape[1]):
            x = i * dx
            y = j * dy
            z = 0.0
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


imgs_path = [os.path.join(DATA_DIR, f"camera_{j:02d}.jpg") for j in range(N_IMAGES)]
print(imgs_path)
imgs = load_images(imgs_path)  # imageio carga en RGB

# Find corners with cv2.findChessboardCornersSB()
corners = [cv2.findChessboardCornersSB(img, CHESSBOARD_SHAPE) for img in imgs]
valid_idx = [i for i, (found, _) in enumerate(corners) if found]
print("Detecciones válidas:", len(valid_idx), "/", len(corners))
for i, (found, _) in enumerate(corners):
    if not found:
        print(f"  Sin tablero en {os.path.basename(imgs_path[i])}: se descarta")
if not valid_idx:
    raise RuntimeError("No se ha detectado el tablero en ninguna imagen")

# No se aplica cornerSubPix: findChessboardCornersSB ya da precisión subpíxel y
# refinar encima desplaza las esquinas ~1 px y empeora el RMS
for i in range(len(imgs)):
    img_bgr = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)  # cv2.imwrite espera BGR
    found, cor = corners[i]
    if found:
        cv2.drawChessboardCorners(img_bgr, CHESSBOARD_SHAPE, cor, found)
    nombre_base = os.path.splitext(os.path.basename(imgs_path[i]))[0]
    write_image(CORNERS_DIR, f"{nombre_base}_esquinas.jpg", img_bgr)

valid_corners = [corners[i][1] for i in valid_idx]
chessboard_points_valid = [get_chessboard_points(CHESSBOARD_SHAPE, SQUARE_SIZE, SQUARE_SIZE)
                           for _ in valid_idx]
image_size = (imgs[0].shape[1], imgs[0].shape[0])  # calibrateCamera espera (ancho, alto)
# Sin distorsión tangencial: con tableros que no cubren todo el encuadre, los
# términos tangenciales se usan para desplazar el centro óptico de forma irreal
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    chessboard_points_valid, valid_corners, image_size, None, None,
    flags=cv2.CALIB_ZERO_TANGENT_DIST)

extrinsics = list(map(lambda rvec, tvec: np.hstack(
    (cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))


print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)

print("\nExtrinsics :")
for i, ext in zip(valid_idx, extrinsics):
    print(f"{os.path.basename(imgs_path[i])}:\n{ext}\n")


np.savez(OUTPUT_CALIB, intrinsics=intrinsics, dist_coeffs=dist_coeffs,
         image_size=np.array(image_size))
print(f"Parámetros de calibración guardados en: {OUTPUT_CALIB}")
