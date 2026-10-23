"""O detector e menos confiante nas plantas DOENTES?

Isto importa muito. Se for, todo limiar de --conf descarta preferencialmente a
classe que o projeto existe para achar — e o erro nao e aleatorio, e enviesado
contra o alvo. Seria o mesmo viés que o indice de vegetacao tinha no estagio 0.

Casa cada caixa rotulada com a deteccao de maior IoU e compara a distribuicao
de confianca entre as caixas doentes e as sadias.

    python tools/conf_doente.py --weights best_v5.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict_tiled import predict_image  # noqa: E402
from tiling import label_path_for, list_images, read_yolo_labels  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, 0, None], b[None, :, 0])
    iy1 = np.maximum(a[:, 1, None], b[None, :, 1])
    ix2 = np.minimum(a[:, 2, None], b[None, :, 2])
    iy2 = np.minimum(a[:, 3, None], b[None, :, 3])
    it = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return it / np.maximum(aa[:, None] + ab[None, :] - it, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best_v5.pt")
    ap.add_argument("--images", type=Path, default=Path("data/raw/images"))
    ap.add_argument("--labels", type=Path, default=Path("data/raw/labels"))
    ap.add_argument("--classe-doente", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="bem baixo de proposito: queremos ver ATE a confianca "
                         "das plantas que um limiar normal descartaria")
    ap.add_argument("--iou-casar", type=float, default=0.4)
    args = ap.parse_args()

    modelo = YOLO(args.weights)
    conf_d: list[float] = []
    conf_s: list[float] = []
    perdidas_d = perdidas_s = 0

    fotos = list_images(args.images)
    for n, p in enumerate(fotos, 1):
        gt = read_yolo_labels(label_path_for(p, args.labels), 5472, 3078)
        if len(gt) == 0:
            continue
        tem_doente = (gt[:, 0] == args.classe_doente).any()
        if not tem_doente:          # so interessa onde ha as duas classes
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        det = predict_image(modelo, img, 1024, 0.4, args.conf, 0.5, 4, 1024,
                            "cpu", descartar_borda=True)
        print(f"[{n}/{len(fotos)}] {p.stem[:28]:30} gt={len(gt):>4} det={len(det):>4}",
              flush=True)
        if len(det) == 0:
            continue
        M = iou_mat(gt[:, 1:5], det[:, 1:5])
        melhor = M.argmax(1)
        valor = M.max(1)
        for i in range(len(gt)):
            eh_doente = gt[i, 0] == args.classe_doente
            if valor[i] < args.iou_casar:
                if eh_doente:
                    perdidas_d += 1
                else:
                    perdidas_s += 1
                continue
            c = float(det[melhor[i], 5])
            (conf_d if eh_doente else conf_s).append(c)

    d, s = np.array(conf_d), np.array(conf_s)
    print(f"\n== Confianca do detector por classe real ==")
    print(f"{'classe':10} {'n':>5} {'p10':>6} {'p25':>6} {'mediana':>8} {'p75':>6}")
    for nome, v in (("doente", d), ("sadia", s)):
        if len(v) == 0:
            print(f"{nome:10} {0:>5}  (nenhuma casada)")
            continue
        print(f"{nome:10} {len(v):>5} {np.percentile(v,10):>6.3f} "
              f"{np.percentile(v,25):>6.3f} {np.median(v):>8.3f} "
              f"{np.percentile(v,75):>6.3f}")

    if len(d) and len(s):
        a = d[:, None]; b = s[None, :]
        auc = float((a > b).mean() + 0.5 * (a == b).mean())
        print(f"\nAUC (doente tem confianca MAIOR que sadia): {auc:.3f}")
        print("  0.50 = sem diferenca | <0.50 = doente e MENOS confiante")

        print(f"\n== Quanto cada limiar descartaria ==")
        print(f"{'conf':>6} {'perde doente':>14} {'perde sadia':>13} {'vies':>8}")
        for t in (0.20, 0.30, 0.40, 0.50):
            pd = float((d < t).mean()) * 100
            ps = float((s < t).mean()) * 100
            print(f"{t:>6.2f} {pd:>13.1f}% {ps:>12.1f}% {pd - ps:>+7.1f}pp")

    print(f"\nnao casadas (IoU < {args.iou_casar}): "
          f"{perdidas_d} doentes, {perdidas_s} sadias")


if __name__ == "__main__":
    main()
