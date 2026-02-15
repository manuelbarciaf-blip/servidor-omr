import json

ANCHO = 1200
ALTO = 1600

NUM_PREGUNTAS = 30  # cambia aquí a 20, 40, 60…

opciones = ["A", "B", "C", "D"]

layout = {
    "zona_respuestas": {
        "x0_rel": 0.12,
        "y0_rel": 0.20,
        "ancho_rel": 0.75,
        "alto_rel": 0.70
    },
    "columnas": {
        "A": {"offset_x_rel": 0.05},
        "B": {"offset_x_rel": 0.27},
        "C": {"offset_x_rel": 0.49},
        "D": {"offset_x_rel": 0.71}
    },
    "filas": {
        "offset_y_rel": 0.0125,
        "alto_celda_rel": 0.028
    },
    "casilla": {
        "ancho_rel": 0.06,
        "alto_rel": 0.028
    }
}

x0 = int(layout["zona_respuestas"]["x0_rel"] * ANCHO)
y0 = int(layout["zona_respuestas"]["y0_rel"] * ALTO)

ancho_casilla = int(layout["casilla"]["ancho_rel"] * ANCHO)
alto_casilla = int(layout["casilla"]["alto_rel"] * ALTO)

offset_y = int(layout["filas"]["offset_y_rel"] * ALTO)

preguntas = []

for i in range(NUM_PREGUNTAS):
    fila_y = y0 + i * offset_y

    opciones_list = []
    for letra in opciones:
        col_offset = layout["columnas"][letra]["offset_x_rel"]
        x1 = x0 + int(col_offset * ANCHO)
        y1 = fila_y
        x2 = x1 + ancho_casilla
        y2 = y1 + alto_casilla

        opciones_list.append({
            "letra": letra,
            "bbox": [x1, y1, x2, y2]
        })

    preguntas.append({
        "numero": i + 1,
        "opciones": opciones_list
    })

plantilla = {
    "umbral": 40,
    "preguntas": preguntas
}

with open("plantilla_layout.json", "w") as f:
    json.dump(plantilla, f, indent=4)

print("Plantilla generada correctamente.")
