"""Diagnostico: mede o tamanho real dos agaves em pixels e recomenda o tile.

Rode ISTO ANTES de tudo. O tamanho do objeto em pixels e o parametro que decide
todo o resto do pipeline, e so voce tem essa informacao (depende da altura do
voo e da camera).

Regra: depois de recortar em tiles de T px e treinar com imgsz=S, o objeto
chega na rede com  obj_px * S / T  pixels. Abaixo de ~24px a deteccao degrada
muito. Entao: T <= obj_px * S / 24.

    python src/inspect_labels.py --images data/raw/images --labels data/raw/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tiling import list_images, read_yolo_labels, label_path_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="imgsz que voce pretende usar no treino")
    ap.add_argument("--min-px", type=float, default=24.0,
                    help="tamanho minimo desejado do objeto na entrada da rede")
    args = ap.parse_args()

    sizes: list[float] = []
    per_image: list[tuple[str, int, int, int, int]] = []

    for img_path in list_images(args.images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [aviso] nao consegui ler {img_path.name}")
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path_for(img_path, args.labels), w, h)
        if len(boxes):
            d = np.sqrt((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2]))
            sizes.extend(d.tolist())
        per_image.append((img_path.name, w, h, len(boxes), int(np.median(
            np.sqrt((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2]))
        )) if len(boxes) else 0))

    print("\n== Imagens ==")
    print(f"{'arquivo':<38} {'largura':>8} {'altura':>7} {'caixas':>7} {'lado_med':>9}")
    for name, w, h, n, med in per_image:
        print(f"{name:<38} {w:>8} {h:>7} {n:>7} {med:>9}")

    if not sizes:
        print("\nNenhuma caixa encontrada. Confira se os .txt estao em --labels "
              "e tem o mesmo nome-base das imagens.")
        return

    arr = np.asarray(sizes)
    print(f"\n== Tamanho dos agaves (lado equivalente, em px da imagem original) ==")
    print(f"  total de instancias : {len(arr)}")
    print(f"  media por imagem    : {len(arr) / max(len(per_image), 1):.1f}")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<2}                 : {np.percentile(arr, q):7.1f} px")

    # Dimensiona pelo p25: queremos que ate as plantas pequenas sobrevivam.
    small = float(np.percentile(arr, 25))
    ideal = small * args.imgsz / args.min_px
    print(f"\n== Recomendacao (imgsz={args.imgsz}, objeto minimo {args.min_px:.0f}px) ==")
    print(f"  tile maximo teorico : {ideal:.0f} px")
    for t in (512, 640, 768, 1024, 1280, 1536, 2048):
        eff = small * args.imgsz / t
        flag = "OK  " if eff >= args.min_px else "RUIM"
        print(f"  tile={t:<5} -> objeto p25 chega com {eff:6.1f} px na rede   [{flag}]")

    # Segunda restricao, e ela manda: o tile precisa ser VARIAS vezes o tamanho
    # da planta. Com tile pouco maior que a planta, quase toda instancia encosta
    # na borda e sai cortada, e o modelo nunca ve a planta inteira com contexto.
    # Abaixo de ~3x isso degrada rapido; 4x a 8x e a faixa saudavel.
    grande = float(np.percentile(arr, 75))
    minimo_ctx = 4 * grande
    print(f"\n  Contexto: a planta grande (p75) tem {grande:.0f}px, entao o tile")
    print(f"  precisa de pelo menos {minimo_ctx:.0f}px para nao cortar tudo na borda.")

    todos = (512, 640, 768, 1024, 1280, 1536, 2048, 2560)
    choices = [t for t in todos
               if small * args.imgsz / t >= args.min_px and t >= minimo_ctx]
    if choices:
        print(f"\n  >> Use --tile {min(choices)}.")
        print(f"     (o menor que satisfaz as DUAS restricoes: objeto grande o")
        print(f"     bastante na rede, e tile grande o bastante para conter a")
        print(f"     planta inteira com folga)")
        cortados = [t for t in todos
                    if small * args.imgsz / t >= args.min_px and t < minimo_ctx]
        if cortados:
            print(f"     Nao use {cortados}: passam no teste de tamanho mas cortam")
            print(f"     as plantas na borda.")
    else:
        print(f"\n  >> Nem tile=512 resolve. Os agaves estao pequenos demais ({small:.0f}px). "
              f"Voe mais baixo, ou aumente o imgsz do treino.")


if __name__ == "__main__":
    main()
