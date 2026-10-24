# Contexto de Imágenes

## Cómo correrlo
```bash
python -m pip install -r requirements.txt
python app.py
```
Imprime un enlace local y uno público (`https://xxxx.gradio.live`, dura ~72 horas).
La primera vez descarga los modelos y calcula los embeddings de la galería
(se guardan en `cache_clip.npy`; si cambias las imágenes de `galeria/` se recalculan solos).

Si sale `operator torchvision::nms does not exist`:
`python -m pip install --upgrade torch torchvision`

## Cambiar el tema de la galería
Reemplaza las imágenes de `galeria/` (mínimo 50, mismo tema) y borra los `cache_*`.
El nombre del archivo se muestra solo al ganar.

## Cómo se calcula el lugar de la imagen secreta (para la demostración)
1. **Embeddings de imagen:** CLIP (`clip-ViT-B-32`) convierte cada foto en un vector de 512 números.
   Se calculan una sola vez para toda la galería y se guardan.
2. **Embeddings de texto:** `clip-ViT-B-32-multilingual-v1` convierte la descripción en un vector
   del *mismo espacio* (fue entrenado para imitar al codificador de texto de CLIP en 50+ idiomas),
   por eso acepta español.
3. **Similitud coseno:** con vectores normalizados (longitud 1) es solo el producto punto
   `galeria_emb @ texto_emb`. Mide el ángulo entre vectores: más cerca de 1 = más parecido.
4. **Lugar:** se ordenan las 60 similitudes de mayor a menor.
   `lugar = 1 + (número de imágenes con similitud mayor que la secreta)`.
   Lugar 1 = ninguna imagen supera a la secreta = victoria.

**Por qué funciona:** CLIP se entrenó con millones de pares (imagen, texto) para que cada foto
quede cerca de su descripción y lejos de las demás. Por eso una frase y una foto se pueden comparar
aunque sean de tipos distintos.

## Extras
- **Pistas:** las 3 imágenes con mayor similitud a tu descripción.
- **Español:** modelo de texto multilingüe.
- **Modo inverso (DINOv2):** subes una foto; se compara con la galería en el espacio de DINOv2
  (`facebook/dinov2-small`, token CLS normalizado), que es imagen-imagen y no entiende texto.

## Créditos de las imágenes
Muestras de ImageNet (repositorio `EliSchwartz/imagenet-sample-images`), 60 razas de perro.
