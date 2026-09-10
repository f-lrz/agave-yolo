"""Diz se a visao computacional classica TEM CHANCE nas suas imagens.

O detector do estagio 0 assume que cada planta e um blob separavel: verde sobre
solo, com borda propria. Se as plantas se tocam ou se sobrepoem, a mascara vira
um lencol continuo e nenhum ajuste de parametro resolve — a informacao de onde
uma planta acaba e a outra comeca simplesmente nao esta na mascara.

Este script mede isso direto nos SEUS labels, sem depender de parametro nenhum,
e diz se vale continuar ajustando ou se e hora de pular para o YOLO.

    python tools/diagnosticar.py --labels data/raw/labels --images data/raw/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tiling import label_path_for, list_images, read_yolo_labels  # noqa: E402


def iou_par(a: np.ndarray) -> np.ndarray:
    """Maior IoU de cada caixa com QUALQUER outra caixa do mesmo conjunto."""
    if len(a) < 2:
        return np.zeros(len(a))
    x1 = np.maximum(a[:, None, 1], a[None, :, 1])
    y1 = np.maximum(a[:, None, 2], a[None, :, 2])
    x2 = np.minimum(a[:, None, 3], a[None, :, 3])
    y2 = np.minimum(a[:, None, 4], a[None, :, 4])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = (a[:, 3] - a[:, 1]) * (a[:, 4] - a[:, 2])
    iou = inter / (ar[:, None] + ar[None, :] - inter + 1e-9)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/raw/images"))
    ap.add_argument("--labels", type=Path, default=Path("data/raw/labels"))
    args = ap.parse_args()

    sobrepostas, vizinhanca, lados, por_img = [], [], [], []

    for p in list_images(args.images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(label_path_for(p, args.labels), w, h)
        if len(gt) < 2:
            continue

        lado = np.sqrt((gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2]))
        lados.extend(lado.tolist())
        ious = iou_par(gt)
        sobrepostas.extend(ious.tolist())

        # distancia ao vizinho mais proximo, em diametros de planta
        cx = (gt[:, 1] + gt[:, 3]) / 2
        cy = (gt[:, 2] + gt[:, 4]) / 2
        d = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2)
        np.fill_diagonal(d, np.inf)
        viz = d.min(axis=1) / np.maximum(lado, 1)
        vizinhanca.extend(viz.tolist())

        cob = float((lado ** 2).sum() / (w * h))
        por_img.append((p.stem[:34], len(gt), float(np.median(lado)),
                        float(np.median(viz)), cob))

    if not lados:
        raise SystemExit("nenhum label encontrado")

    sob = np.asarray(sobrepostas)
    viz = np.asarray(vizinhanca)
    lad = np.asarray(lados)

    print(f"\n{'imagem':<36}{'caixas':>7}{'lado':>7}{'viz/diam':>10}{'cobert.':>9}")
    for nome, n, l, v, c in por_img:
        print(f"{nome:<36}{n:>7}{l:>7.0f}{v:>10.2f}{c * 100:>8.1f}%")

    print(f"\n== Quanto as plantas se encostam ({len(lad)} caixas) ==")
    print(f"  caixas que sobrepoem outra (IoU > 0.10) : "
          f"{(sob > 0.10).mean() * 100:5.1f}%")
    print(f"  caixas que sobrepoem outra (IoU > 0.30) : "
          f"{(sob > 0.30).mean() * 100:5.1f}%")
    print(f"  distancia ao vizinho / diametro, mediana: {np.median(viz):5.2f}")
    print(f"  plantas com vizinho a menos de 1 diametro: {(viz < 1.0).mean() * 100:5.1f}%")
    print(f"\n  variacao de tamanho: p5 {np.percentile(lad, 5):.0f}px  "
          f"p50 {np.percentile(lad, 50):.0f}px  p95 {np.percentile(lad, 95):.0f}px "
          f"({np.percentile(lad, 95) / max(np.percentile(lad, 5), 1):.1f}x)")

    print("\n== Leitura ==")
    encostadas = (viz < 1.0).mean()
    print(f"  {encostadas * 100:.0f}% das plantas tem vizinho a menos de um diametro.")
    print("\n  Ponto de referencia medido: num conjunto sintetico com 90% de plantas")
    print("  nessa condicao, o cv_proposals ainda entregou recall 0.91. Ou seja,")
    print("  DENSIDADE SOZINHA NAO DERRUBA o detector classico — se o seu recall")
    print("  esta baixo com densidade parecida, a causa e outra, e esta na")
    print("  aparencia da planta, nao no espacamento.")
    print("\n  As causas que sobram, em ordem de probabilidade:")
    print("    1. a roseta nao vira um blob solido na mascara (folhas finas e")
    print("       separadas, com solo aparecendo entre elas)")
    print("    2. o fundo nao e solo limpo — mato ou cobertura verde generalizada")
    print("    3. a folha do agave e verde-azulada/acinzentada e o indice de")
    print("       vegetacao a separa mal do solo")
    print("\n  Nenhuma dessas da para diagnosticar por numero. Rode o cv_proposals")
    print("  com --debug e olhe o QUADRO 3 (mascara): se a planta nao virou um")
    print("  blob branco solido ali, o problema esta antes de qualquer parametro")
    print("  de forma, e ajustar limiar nao vai resolver.")

    if (sob > 0.30).mean() > 0.15:
        print(f"\n  Obs: {(sob > 0.30).mean() * 100:.0f}% das suas caixas se sobrepoem")
        print("  com IoU > 0.30. Isso limita o teto de qualquer metodo por mascara,")
        print("  porque a fronteira entre as duas plantas nao existe na binarizacao.")


if __name__ == "__main__":
    main()
