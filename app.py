"""
Contexto de Imágenes
====================
La computadora elige en secreto una imagen de la galería (`galeria/`).
El jugador escribe descripciones; por cada una, el juego ordena TODA la galería
de la imagen más parecida a la menos parecida y dice en qué lugar quedó la
imagen secreta. Se gana al ponerla en el lugar 1.

Cómo funciona (resumen para la demostración)
--------------------------------------------
1. Un modelo CLIP convierte cada imagen en un vector (embedding). Los de la
   galería se calculan UNA sola vez y se guardan en disco (cache_*.npy).
2. Un modelo de texto multilingüe, entrenado para caer en el MISMO espacio que
   CLIP, convierte la descripción en un vector (en español o inglés).
3. Como ambos viven en el mismo espacio, se comparan con similitud coseno
   (producto punto de vectores normalizados).
4. Se ordenan las imágenes por similitud. El lugar de la secreta es
   1 + (cuántas imágenes tienen similitud MAYOR que la secreta).

Extras incluidos
----------------
- Pistas: muestra las 3 imágenes que mejor corresponden a la descripción.
- En español: modelo de texto multilingüe (clip-ViT-B-32-multilingual-v1).
- Modo inverso con DINOv2: el jugador sube una foto y se compara con la
  secreta en el espacio de DINOv2 (imagen-imagen).

Uso:  python -m pip install -r requirements.txt
      python app.py          (imprime un enlace público válido ~72 horas)
"""
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

GALERIA = Path("galeria")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MODELO_IMAGEN = "clip-ViT-B-32"
MODELO_TEXTO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # alineado con clip-ViT-B-32
MODELO_DINO = "facebook/dinov2-small"
N_PISTAS = 3


# --------------------------------------------------------------------------
# Galería y caché de embeddings
# --------------------------------------------------------------------------
def archivos_galeria():
    return sorted(p for p in GALERIA.iterdir() if p.suffix.lower() in EXTS)


def firma_galeria(archivos):
    """Huella de la galería: si cambia, los embeddings guardados se recalculan."""
    texto = json.dumps([(p.name, p.stat().st_size) for p in archivos])
    return hashlib.md5(texto.encode()).hexdigest()


def cargar_o_calcular(nombre_cache, calcular):
    archivos = archivos_galeria()
    firma = firma_galeria(archivos)
    npy, meta = Path(f"cache_{nombre_cache}.npy"), Path(f"cache_{nombre_cache}.json")
    if npy.exists() and meta.exists() and json.loads(meta.read_text())["firma"] == firma:
        return np.load(npy)
    emb = calcular([Image.open(p).convert("RGB") for p in archivos])
    np.save(npy, emb)
    meta.write_text(json.dumps({"firma": firma}))
    return emb


# --------------------------------------------------------------------------
# Modelos (se cargan la primera vez que se usan)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def modelo_imagen():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO_IMAGEN)


@lru_cache(maxsize=None)
def modelo_texto():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO_TEXTO)


@lru_cache(maxsize=None)
def modelo_dino():
    from transformers import AutoImageProcessor, AutoModel
    return AutoImageProcessor.from_pretrained(MODELO_DINO), AutoModel.from_pretrained(MODELO_DINO).eval()


def emb_clip_imagenes(imagenes):
    return modelo_imagen().encode(imagenes, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)


def emb_clip_texto(texto):
    return modelo_texto().encode([texto], normalize_embeddings=True, convert_to_numpy=True)[0]


def emb_dino(imagenes):
    import torch
    proc, modelo = modelo_dino()
    salida = []
    for i in range(0, len(imagenes), 16):
        entradas = proc(images=imagenes[i:i + 16], return_tensors="pt")
        with torch.no_grad():
            cls = modelo(**entradas).last_hidden_state[:, 0]  # token CLS = resumen de la imagen
        salida.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
    return np.concatenate(salida)


def galeria_clip():
    return cargar_o_calcular("clip", emb_clip_imagenes)


def galeria_dino():
    return cargar_o_calcular("dino", emb_dino)


# --------------------------------------------------------------------------
# Lógica del juego
# --------------------------------------------------------------------------
def lugar_de(similitudes: np.ndarray, secreta: int) -> int:
    """Lugar (1 = más parecida) de la imagen secreta en la galería ordenada."""
    return int((similitudes > similitudes[secreta]).sum()) + 1


def partida_nueva():
    return {"secreta": random.randrange(len(archivos_galeria())), "historial": [], "gano": False, "rendido": False}


def terminada(estado):
    return estado["gano"] or estado["rendido"]


def nombre_secreta(estado):
    return archivos_galeria()[estado["secreta"]].stem.replace("_", " ")


def revelar(estado):
    """Muestra la imagen secreta (al ganar o al rendirse)."""
    return gr.update(value=str(archivos_galeria()[estado["secreta"]]), visible=True)


def tabla(estado):
    # Del mejor lugar al peor, como en Contexto
    filas = sorted(estado["historial"], key=lambda f: f[2])
    return [[tipo, intento, lugar] for tipo, intento, lugar in filas]


def mensaje_inicial():
    n = len(archivos_galeria())
    return (f"### Hay una imagen secreta entre {n} fotos de perros.\n"
            "Describe lo que ves (color, pelaje, orejas, tamaño, fondo...) en español o inglés. "
            "Te diré en qué **lugar** quedó la imagen secreta: **1 = ¡ganaste!**")


def iniciar():
    estado = partida_nueva()
    return estado, mensaje_inicial(), [], [], None, gr.update(value=None, visible=False)


def reiniciar():
    return iniciar()


def rendirse(estado):
    estado = estado or partida_nueva()
    if terminada(estado):
        return estado, gr.update(), gr.update()
    estado["rendido"] = True
    msg = (f"## 🏳️ Te rendiste tras {len(estado['historial'])} intentos.\n"
           f"La imagen secreta era un **{nombre_secreta(estado)}**.")
    return estado, msg, revelar(estado)


def procesar(estado, tipo, intento_txt, similitudes):
    """Guarda el intento, calcula el mensaje y devuelve (estado, mensaje, tabla)."""
    lugar = lugar_de(similitudes, estado["secreta"])
    estado["historial"].append((tipo, intento_txt, lugar))
    if lugar == 1:
        estado["gano"] = True
        msg = f"## 🎉 ¡Ganaste en {len(estado['historial'])} intentos!\nLa imagen era un **{nombre_secreta(estado)}**."
    else:
        msg = f"### La imagen secreta quedó en el lugar **{lugar}** de {len(similitudes)}."
    return estado, msg, tabla(estado)


def adivinar_texto(estado, descripcion):
    estado = estado or partida_nueva()
    descripcion = (descripcion or "").strip()
    if terminada(estado) or not descripcion:
        return estado, gr.update(), tabla(estado), gr.update(), "", gr.update()
    sims = galeria_clip() @ emb_clip_texto(descripcion)
    estado, msg, filas = procesar(estado, "texto", descripcion, sims)
    archivos = archivos_galeria()
    top = np.argsort(-sims)[:N_PISTAS]  # pista: las que mejor corresponden a la descripción
    pistas = [(Image.open(archivos[i]).convert("RGB"), f"Lugar {k + 1}") for k, i in enumerate(top)]
    return estado, msg, filas, pistas, "", (revelar(estado) if estado["gano"] else gr.update())


def adivinar_foto(estado, foto):
    estado = estado or partida_nueva()
    if terminada(estado) or foto is None:
        return estado, gr.update(), tabla(estado), gr.update()
    emb = emb_dino([foto.convert("RGB")])[0]
    sims = galeria_dino() @ emb
    estado, msg, filas = procesar(estado, "foto", "(foto subida)", sims)
    return estado, msg, filas, (revelar(estado) if estado["gano"] else gr.update())


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
with gr.Blocks(title="Contexto de Imágenes") as demo:
    estado = gr.State(None)  # una partida independiente por jugador
    gr.Markdown("# 🖼️ Contexto de Imágenes")
    mensaje = gr.Markdown()

    with gr.Tabs():
        with gr.Tab("Describir con texto"):
            descripcion = gr.Textbox(label="Tu descripción", placeholder="p. ej. perro café con orejas largas en el pasto")
            boton_texto = gr.Button("Probar", variant="primary")
            pistas = gr.Gallery(label=f"Pista: las {N_PISTAS} imágenes que mejor corresponden a tu descripción",
                                columns=N_PISTAS, height=220)
        with gr.Tab("Modo inverso (subir foto)"):
            foto = gr.Image(type="pil", label="Sube una foto de un perro")
            boton_foto = gr.Button("Probar con esta foto", variant="primary")

    historial = gr.Dataframe(headers=["Modo", "Intento", "Lugar"], datatype=["str", "str", "number"],
                             label="Historial (mejor lugar primero)", interactive=False)
    revelada = gr.Image(label="Imagen secreta", visible=False, interactive=False)
    with gr.Row():
        boton_rendir = gr.Button("🏳️ Rendirme")
        boton_nueva = gr.Button("🔄 Nueva partida")

    salidas_inicio = [estado, mensaje, historial, pistas, foto, revelada]
    demo.load(iniciar, None, salidas_inicio)
    boton_nueva.click(reiniciar, None, salidas_inicio)
    boton_rendir.click(rendirse, estado, [estado, mensaje, revelada])

    for disparador in (boton_texto.click, descripcion.submit):
        disparador(adivinar_texto, [estado, descripcion], [estado, mensaje, historial, pistas, descripcion, revelada])
    boton_foto.click(adivinar_foto, [estado, foto], [estado, mensaje, historial, revelada])

if __name__ == "__main__":
    n = len(archivos_galeria())
    assert n >= 50, f"La galería tiene {n} imágenes; se necesitan al menos 50."
    print("Calculando embeddings de la galería (solo la primera vez)...")
    galeria_clip()
    demo.launch(share=True)  # share=True genera un enlace público que dura 72 horas
