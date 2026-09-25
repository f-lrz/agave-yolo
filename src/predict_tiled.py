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
                  device: str, descartar_borda: bool = False,
                  margem: int = 2) -> np.ndarray:
    """Retorna array (N, 6): [cls, x1, y1, x2, y2, conf] na imagem inteira."""
    h, w = img.shape[:2]
    grid = list(tile_grid(w, h, tile, overlap))
    found: list[np.ndarray] = []

    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]
        results = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, verbose=False)
        for (x0, y0, x1, y1), r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()[:, None]
            cf = r.boxes.conf.cpu().numpy()[:, None]

            if descartar_borda:
                # Uma caixa encostada na borda do tile e quase sempre um PEDACO
                # de planta, nao uma planta. A sobreposicao garante que ela
                # aparece inteira no tile vizinho, entao descartar aqui nao
                # perde a deteccao — so a versao truncada dela.
                # Bordas que sao a borda da IMAGEM nao contam: ali nao existe
                # tile vizinho para recuperar a planta.
                tw, th = x1 - x0, y1 - y0
                na_borda = (((xyxy[:, 0] <= margem) & (x0 > 0))
                            | ((xyxy[:, 1] <= margem) & (y0 > 0))
                            | ((xyxy[:, 2] >= tw - margem) & (x1 < w))
                            | ((xyxy[:, 3] >= th - margem) & (y1 < h)))
                manter = ~na_borda
                if not manter.any():
                    continue
                xyxy, cls, cf = xyxy[manter], cls[manter], cf[manter]

            xyxy[:, [0, 2]] += x0          # tile -> imagem inteira
            xyxy[:, [1, 3]] += y0
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


# BGR (nao RGB — e OpenCV). Verde confunde com a propria planta; preto e o
# padrao porque contrasta tanto com a folha verde quanto com o capim seco.
CORES = {
    "preto": (0, 0, 0),
    "branco": (255, 255, 255),
    "magenta": (255, 0, 255),
    "ciano": (255, 255, 0),
    "vermelho": (0, 0, 255),
    "verde": (0, 200, 0),
}


def draw(img: np.ndarray, det: np.ndarray, names: dict[int, str],
         cor: tuple[int, int, int] = (0, 0, 0), halo: bool = True) -> np.ndarray:
    out = img.copy()
    thick = max(1, round(min(img.shape[:2]) / 900))
    # Um contorno claro por baixo mantem a caixa visivel tambem quando ela cai
    # sobre sombra ou solo escuro, onde o preto puro sumiria.
    cor_halo = (255, 255, 255) if sum(cor) < 384 else (0, 0, 0)
    for c, x1, y1, x2, y2, cf in det:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if halo:
            cv2.rectangle(out, p1, p2, cor_halo, thick + 2)
        cv2.rectangle(out, p1, p2, cor, thick)
        label = f"{names.get(int(c), int(c))} {cf:.2f}"
        org = (int(x1), max(int(y1) - 4, 12))
        if halo:
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * thick, cor_halo, thick + 2)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thick, cor, thick)
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
    ap.add_argument("--cor", choices=list(CORES), default="preto",
                    help="cor das caixas no preview (padrao: preto)")
    ap.add_argument("--sem-halo", action="store_true",
                    help="desliga o contorno claro por tras da caixa")
    ap.add_argument("--descartar-borda", action="store_true",
                    help="descarta deteccao colada na borda do tile: e quase "
                         "sempre um PEDACO de planta, e a sobreposicao garante "
                         "que ela aparece inteira no tile vizinho")
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
                            args.iou, args.batch, args.imgsz, args.device,
                            descartar_borda=args.descartar_borda)
        total += len(det)
        if not args.no_preview:
            cv2.imwrite(str(args.out / "images" / f"{p.stem}.jpg"),
                        draw(img, det, names, cor=CORES[args.cor],
                             halo=not args.sem_halo),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
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
