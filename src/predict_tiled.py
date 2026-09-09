"""Inferencia em imagem inteira de drone: recorta em tiles, prediz, remonta.

E o mesmo script que gera as PRE-ANOTACOES do loop de rotulagem: com
--save-labels ele escreve .txt YOLO nas coordenadas da imagem original, prontos
para voce abrir no CVAT/LabelImg/Roboflow e apenas CORRIGIR em vez de desenhar
tudo do zero.

    # visualizar
    python src/predict_tiled.py --weights runs/detect/agave_seed/weights/best.pt \
        --images data/raw/nao_rotuladas --out out/preview --tile 1024

    # gerar pre-anotacoes para corrigir
    python src/predict_tiled.py --weights ... --images ... --out data/preanot \
        --save-labels --conf 0.20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from tqdm import tqdm
from ultralytics import YOLO

from tiling import list_images, tile_grid, to_yolo_lines


def predict_image(model: YOLO, img: np.ndarray, tile: int, overlap: float,
                  conf: float, iou: float, batch: int, imgsz: int,
                  device: str) -> np.ndarray:
    """Retorna array (N, 6): [cls, x1, y1, x2, y2, conf] na imagem inteira."""
    h, w = img.shape[:2]
    grid = list(tile_grid(w, h, tile, overlap))
    found: list[np.ndarray] = []

    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]
        results = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, verbose=False)
        for (x0, y0, _, _), r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            xyxy[:, [0, 2]] += x0          # tile -> imagem inteira
            xyxy[:, [1, 3]] += y0
            cls = r.boxes.cls.cpu().numpy()[:, None]
            cf = r.boxes.conf.cpu().numpy()[:, None]
            found.append(np.hstack([cls, xyxy, cf]))

    if not found:
        return np.zeros((0, 6))

    det = np.vstack(found)
    # NMS global: a mesma planta aparece em ate 4 tiles por causa da sobreposicao
    keep_all: list[np.ndarray] = []
    for c in np.unique(det[:, 0]):
        d = det[det[:, 0] == c]
        idx = nms(torch.from_numpy(d[:, 1:5]).float(),
                  torch.from_numpy(d[:, 5]).float(), iou).numpy()
        keep_all.append(d[idx])
    det = np.vstack(keep_all)
    return det[np.argsort(-det[:, 5])]


def draw(img: np.ndarray, det: np.ndarray, names: dict[int, str],
         colors=((0, 200, 0), (0, 0, 255), (255, 160, 0))) -> np.ndarray:
    out = img.copy()
    thick = max(1, round(min(img.shape[:2]) / 900))
    for c, x1, y1, x2, y2, cf in det:
        color = colors[int(c) % len(colors)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        label = f"{names.get(int(c), int(c))} {cf:.2f}"
        cv2.putText(out, label, (int(x1), max(int(y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color, thick)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4, help="tiles por lote (6GB: 4)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--classe", type=int, default=None,
                    help="reescreve o id de classe nos .txt salvos; use quando o "
                         "detector e de 1 classe mas seu projeto de anotacao tem outras")
    ap.add_argument("--save-labels", action="store_true",
                    help="escreve .txt YOLO para o loop de correcao")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        (args.out / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    for p in tqdm(list_images(args.images), desc="inferencia"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        det = predict_image(model, img, args.tile, args.overlap, args.conf,
                            args.iou, args.batch, args.imgsz, args.device)
        total += len(det)
        if not args.no_preview:
            cv2.imwrite(str(args.out / "images" / f"{p.stem}.jpg"),
                        draw(img, det, names), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.save_labels:
            saida = det[:, :5].copy()
            if args.classe is not None:
                saida[:, 0] = args.classe      # casa com a convencao do seu projeto
            (args.out / "labels" / f"{p.stem}.txt").write_text(
                "\n".join(to_yolo_lines(saida, w, h)) + "\n")

    print(f"\n{total} deteccoes -> {args.out}")
    if args.save_labels:
        print("Abra data/raw/<imagens> + estes .txt no CVAT/LabelImg e CORRIJA "
              "(apagar falso positivo, desenhar o que faltou). Depois retreine.")


if __name__ == "__main__":
    main()
