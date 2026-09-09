"""Recorta as imagens brutas (+ labels) em um dataset YOLO de tiles.

Ponto critico: o split treino/val e feito POR IMAGEM DE ORIGEM, nunca por tile.
Tiles vizinhos se sobrepoem — se voce embaralhar tiles, pedacos da mesma planta
caem em treino e em validacao, sua mAP fica linda e o modelo e inutil em campo.

    python src/make_tiles.py --images data/raw/images --labels data/raw/labels \
        --out data/tiles --tile 1024 --overlap 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tiling import (boxes_for_tile, label_path_for, list_images,
                    read_yolo_labels, tile_grid, to_yolo_lines)


def build_split(stems: list[str], ratios: tuple[float, float, float],
                seed: int) -> dict[str, str]:
    """Distribui as imagens de origem entre train/val/test."""
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = max(1, round(n * ratios[0]))
    n_val = max(1, round(n * ratios[1])) if n - n_train >= 1 else 0
    # com poucas imagens, garante que sobre pelo menos 1 para treino
    n_train = min(n_train, n - n_val)
    assign = {}
    for i, s in enumerate(stems):
        assign[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visibility", type=float, default=0.3,
                    help="fracao minima da caixa que precisa sobrar no tile")
    ap.add_argument("--empty-ratio", type=float, default=0.1,
                    help="fracao dos tiles sem nenhum agave que e mantida "
                         "(negativos ajudam a reduzir falsos positivos, mas em "
                         "excesso desbalanceiam o treino)")
    ap.add_argument("--val-images", nargs="*", default=None,
                    help="nomes-base das imagens que devem ir para validacao "
                         "(sobrepoe o split aleatorio; use para separar por talhao/voo)")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="*", default=["agave"])
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit(f"nenhuma imagem em {args.images}")

    if args.val_images:
        val = set(args.val_images)
        assign = {p.stem: ("val" if p.stem in val else "train") for p in imgs}
    else:
        assign = build_split([p.stem for p in imgs], tuple(args.ratios), args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    for sp in ("train", "val", "test"):
        (args.out / sp / "images").mkdir(parents=True, exist_ok=True)
        (args.out / sp / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = {sp: [0, 0, 0] for sp in ("train", "val", "test")}  # tiles, tiles c/ obj, caixas

    for img_path in tqdm(imgs, desc="tiles"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[aviso] ilegivel: {img_path.name}")
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path_for(img_path, args.labels), w, h)
        sp = assign[img_path.stem]

        for x0, y0, x1, y1 in tile_grid(w, h, args.tile, args.overlap):
            tb = boxes_for_tile(boxes, (x0, y0, x1, y1), args.min_visibility)
            # tiles vazios entram so na proporcao pedida, e nunca na validacao
            # (senao a mAP fica dominada por imagens sem objeto)
            if len(tb) == 0 and (sp != "train" or rng.random() > args.empty_ratio):
                continue
            crop = img[y0:y1, x0:x1]
            th, tw = crop.shape[:2]
            name = f"{img_path.stem}__x{x0}_y{y0}"
            cv2.imwrite(str(args.out / sp / "images" / f"{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            lines = to_yolo_lines(tb, tw, th)
            (args.out / sp / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            stats[sp][0] += 1
            stats[sp][1] += 1 if lines else 0
            stats[sp][2] += len(lines)

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(args.names))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        f"train: train/images\nval: val/images\ntest: test/images\n\nnames:\n{names_yaml}\n"
    )

    print("\n== Dataset gerado ==")
    for sp in ("train", "val", "test"):
        origem = sorted(s for s, v in assign.items() if v == sp)
        t, tp, b = stats[sp]
        print(f"  {sp:<5} {t:>5} tiles ({tp} com objeto, {b} caixas)  <- {len(origem)} imagens")
        for s in origem:
            print(f"           . {s}")
    print(f"\n  data.yaml -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
