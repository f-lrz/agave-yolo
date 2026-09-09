"""ETAPA 2 — extrai um recorte por agave detectado/rotulado.

Aqui esta o grande ganho de reaproveitar a etapa 1: voce NAO desenha caixa
nenhuma de novo. Cada caixa da etapa 1 vira um JPG de 224x224 numa pasta
`_classificar/`, e rotular vira arrastar arquivo entre duas pastas — coisa que
da para fazer no Explorer do Windows, centenas por hora, e que um agronomo
consegue revisar sem aprender ferramenta de anotacao.

    python src/make_crops.py --images data/raw/images --labels data/raw/labels \
        --out data/crops

Depois: mova os arquivos de data/crops/_classificar/ para
  data/crops/train/sadia/  e  data/crops/train/doente/
(e separe ~20% em data/crops/val/sadia|doente — de imagens de origem diferentes).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from tiling import label_path_for, list_images, read_yolo_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="margem extra ao redor da caixa; um pouco de solo e "
                         "vizinhanca ajuda o classificador a julgar o vigor")
    ap.add_argument("--min-px", type=int, default=24,
                    help="descarta recortes menores que isso (irreconheciveis)")
    args = ap.parse_args()

    dest = args.out / "_classificar"
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for cls in ("sadia", "doente"):
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)

    index = [("crop", "imagem_origem", "x1", "y1", "x2", "y2")]
    n = skipped = 0

    for p in tqdm(list_images(args.images), desc="recortes"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path_for(p, args.labels), w, h)
        for i, (_, x1, y1, x2, y2) in enumerate(boxes):
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < args.min_px:
                skipped += 1
                continue
            mx, my = bw * args.pad, bh * args.pad
            cx1 = max(int(x1 - mx), 0)
            cy1 = max(int(y1 - my), 0)
            cx2 = min(int(x2 + mx), w)
            cy2 = min(int(y2 + my), h)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                skipped += 1
                continue
            crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
            name = f"{p.stem}__{i:04d}.jpg"
            cv2.imwrite(str(dest / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            index.append((name, p.name, cx1, cy1, cx2, cy2))
            n += 1

    with (args.out / "crops_index.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(index)

    print(f"\n{n} recortes em {dest}  ({skipped} descartados por serem pequenos demais)")
    print("Agora e so arrastar os arquivos para train/sadia, train/doente "
          "(e ~20% para val/, de imagens de origem diferentes das do train).")
    print(f"O mapa recorte -> caixa original esta em {args.out / 'crops_index.csv'}")


if __name__ == "__main__":
    main()
