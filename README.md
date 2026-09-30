# Sistema de seguridad visual y tracking en tiempo real

Proyecto de Visión por Ordenador que implementa, sobre vídeo en directo, un sistema de desbloqueo mediante una secuencia de figuras de colores y un minijuego de billar basado en el seguimiento de una bola. Todo el sistema utiliza técnicas clásicas de visión por computador con OpenCV, sin redes neuronales.

## Objetivo

A partir de la imagen de una cámara, el proyecto resuelve tres problemas encadenados:

1. **¿Cómo corregir la distorsión que introduce la lente de la cámara?** (calibración geométrica)
2. **¿Es posible usar una secuencia de figuras físicas como contraseña?** (detección de patrones por color y forma, y decodificación de secuencias)
3. **¿Se puede seguir un objeto en movimiento de forma robusta y reaccionar a lo que ocurre con él?** (tracking y lógica de juego)

## Funcionamiento

El sistema tiene dos fases que se ejecutan en la misma ventana:

- **Login de seguridad.** Hay que mostrar ante la cámara, dentro de una región marcada en pantalla, cuatro figuras en orden: triángulo rojo, cuadrado verde, círculo rojo y rombo azul. Si aparece una figura incorrecta, la secuencia se reinicia.
- **Juego de billar.** Tras el desbloqueo, el usuario marca las zonas que actúan como agujeros y selecciona la bola azul. El sistema la sigue en tiempo real y suma un punto cuando la bola desaparece dentro de un agujero.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `Lab_Project/src/calibration.py` | Calibración de la cámara a partir de fotos de un tablero de ajedrez |
| `Lab_Project/src/main.py` | Aplicación principal: captura de vídeo, login de seguridad y juego |
| `Lab_Project/data/` | Imágenes de calibración y parámetros calculados (`camera_calibration_params.npz`) |
| `Lab_Project/esquinas/` | Imágenes de calibración con las esquinas detectadas, para verificación visual |
| `Lab_Project/template/` | Enunciado de la práctica y plantilla LaTeX de la memoria |
| `requirements.txt` | Dependencias del proyecto |

## Enfoque técnico

- **Calibración**: detección de las 49 esquinas interiores de un tablero de ajedrez en 9 imágenes con `findChessboardCornersSB`, que ofrece precisión subpíxel, y estimación de los parámetros intrínsecos y de distorsión radial con `calibrateCamera` (error de reproyección ≈ 1,2 px). Los términos de distorsión tangencial se fijan a cero para evitar estimaciones irreales del centro óptico.
- **Corrección de distorsión**: los parámetros intrínsecos se reescalan a la resolución real del vídeo, de modo que la calibración es válida aunque se haya realizado a otra resolución. La corrección se precalcula con `initUndistortRectifyMap` y se aplica a cada fotograma con `remap`, más eficiente que corregir cada imagen desde cero.
- **Detección de patrones**: segmentación por color en el espacio HSV, limpieza de las máscaras con operaciones morfológicas y clasificación de la forma a partir del contorno (aproximación poligonal para contar vértices, circularidad para los círculos y orientación del rectángulo mínimo para distinguir cuadrados de rombos).
- **Decodificación de la secuencia**: una figura solo se acepta si se detecta durante varios fotogramas consecutivos, lo que filtra falsos positivos puntuales. Cualquier figura fuera de orden reinicia la secuencia.
- **Tracking**: la bola se busca únicamente en una ventana alrededor de su última posición y se descartan saltos físicamente imposibles entre fotogramas, además de las manchas azules que no son circulares. Si la bola se pierde, el sistema tolera unos fotogramas de ausencia y después la vuelve a buscar cerca de donde desapareció.
- **Lógica de puntuación**: se suma un punto cuando la bola desaparece tras haber permanecido mayoritariamente dentro de un agujero en los últimos fotogramas. Si reaparece en el mismo agujero poco después (por ejemplo, porque se ha tapado con la mano), el punto se anula.

## Tecnologías

Python · OpenCV · NumPy · imageio

## Cómo ejecutarlo

Se necesita una webcam conectada al ordenador.

1. Clona el repositorio:
   ```bash
   git clone https://github.com/jsarabiag/proyecto_vision.git
   cd proyecto_vision
   ```
2. Crea un entorno e instala las dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # En macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. *(Opcional)* Recalcula la calibración. El repositorio ya incluye los parámetros calculados, pero corresponden a la cámara original; para calibrar otra cámara, sustituye las fotos de `Lab_Project/data/` por fotos propias de un tablero de 8×8 casillas.
   ```bash
   python Lab_Project/src/calibration.py
   ```
4. Lanza la aplicación:
   ```bash
   python Lab_Project/src/main.py
   ```

### Controles del juego

| Tecla | Acción |
|---|---|
| `h` | Marcar un agujero dibujando un rectángulo con el ratón |
| `b` | Seleccionar la bola haciendo clic sobre ella |
| `r` | Reiniciar la puntuación |
| `q` / `Esc` | Salir |

## Limitaciones

- La detección se basa en rangos de color fijos, por lo que es sensible a las condiciones de iluminación y a la presencia de objetos de colores similares en el fondo.
- Los parámetros de calibración incluidos solo son exactos para la cámara con la que se tomaron las imágenes.

---
*Proyecto desarrollado por Iñaki Juan-Aracil y Javier Sarabia García como parte de la asignatura de Visión por Ordenador — Grado en Ingeniería Matemática e Inteligencia Artificial (ICAI, Universidad Pontificia Comillas).*