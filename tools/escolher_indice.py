"""Descobre QUAL indice de vegetacao separa agave do fundo nas SUAS imagens.

O Excess Green assume fundo de solo nu. Se o seu terreno tem capim seco, mato
rasteiro ou cobertura verde, o ExG marca o fundo inteiro como vegetacao e a
mascara vira um lencol continuo — a partir dai nenhum ajuste de limiar, tamanho
ou forma resolve, porque a informacao de onde esta a planta ja se perdeu.

Este script usa as caixas que voce ja rotulou para medir, indice por indice,
o quanto os pixels DE DENTRO das plantas se separam dos pixels do fundo. Nao
depende de nenhum parametro do detector.

    python tools/escolher_indice.py --images data/raw/images --labels data/raw/labels

Le como AUC: 1.00 = separacao perfeita, 0.50 = o indice nao distingue nada.
Abaixo de ~0.80 nenhum ajuste posterior salva.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cv_proposals import vegetation_index                          # noqa: E402
from tiling import label_path_for, list_images, read_yolo_labels   # noqa: E402

INDICES = ["exg", "exgr", "vari", "gli", "hsv", "lab_a", "verde_escuro"]


def auc_por_histograma(hp: np.ndarray, hn: np.ndarray) -> float:
    """AUC exata a partir dos histogramas das duas classes (256 bins)."""
    np_, nn = hp.sum(), hn.sum()
    if np_ == 0 or nn == 0:
        return 0.5
    menores = np.concatenate([[0], np.cumsum(hn)[:-1]])     # fundo abaixo de v
    return float((hp * (menores + 0.5 * hn)).sum() / (np_ * nn))


def melhor_corte(hp: np.ndarray, hn: np.ndarray) -> tuple[int, float, float]:
    """Limiar que maximiza (sensibilidade - falso positivo). -> (corte, TPR, FPR)"""
    tpr = 1.0 - np.concatenate([[0], np.cumsum(hp)[:-1]]) / max(hp.sum(), 1)
    fpr = 1.0 - np.concatenate([[0], np.cumsum(hn)[:-1]]) / max(hn.sum(), 1)
    j = tpr - fpr
    t = int(np.argmax(j))
    return t, float(tpr[t]), float(fpr[t])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/raw/images"))
    ap.add_argument("--labels", type=Path, default=Path("data/raw/labels"))
    ap.add_argument("--max-lado", type=int, default=1600)
    ap.add_argument("--margem", type=float, default=0.5,
                    help="fracao central da caixa usada como 'planta'; o miolo "
                         "evita pegar fundo que sobra nos cantos da caixa")
    args = ap.parse_args()

    hists = {k: [np.zeros(256, np.int64), np.zeros(256, np.int64)] for k in INDICES}
    otsus = {k: [] for k in INDICES}
    n_img = 0

    for p in list_images(args.images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        gt = read_yolo_labels(label_path_for(p, args.labels), w0, h0)
        if len(gt) == 0:
            continue

        esc = min(1.0, args.max_lado / max(h0, w0))
        if esc < 1.0:
            img = cv2.resize(img, (int(w0 * esc), int(h0 * esc)),
                             interpolation=cv2.INTER_AREA)
            gt = gt.copy()
            gt[:, 1:5] *= esc
        h, w = img.shape[:2]

        # mascara "planta" = miolo das caixas; "fundo" = longe de qualquer caixa
        planta = np.zeros((h, w), np.uint8)
        perto = np.zeros((h, w), np.uint8)
        for _, x1, y1, x2, y2 in gt:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            mw, mh = (x2 - x1) * args.margem / 2, (y2 - y1) * args.margem / 2
            cv2.rectangle(planta, (int(cx - mw), int(cy - mh)),
                          (int(cx + mw), int(cy + mh)), 255, -1)
            cv2.rectangle(perto, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        # dilata a zona proibida: o fundo tem que ser fundo de verdade
        perto = cv2.dilate(perto, np.ones((15, 15), np.uint8))
        fundo = cv2.bitwise_not(perto)

        if planta.sum() == 0 or fundo.sum() == 0:
            continue
        n_img += 1
        for k in INDICES:
            veg = vegetation_index(img, k)
            hists[k][0] += cv2.calcHist([veg], [0], planta, [256], [0, 256]).ravel().astype(np.int64)
            hists[k][1] += cv2.calcHist([veg], [0], fundo, [256], [0, 256]).ravel().astype(np.int64)
            otsu, _ = cv2.threshold(veg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsus[k].append(otsu)

    if n_img == 0:
        raise SystemExit("nenhuma imagem com labels utilizavel")

    linhas = []
    for k in INDICES:
        hp, hn = hists[k]
        auc = auc_por_histograma(hp, hn)
        corte, tpr, fpr = melhor_corte(hp, hn)
        otsu = float(np.median(otsus[k]))
        linhas.append((auc, k, corte, tpr, fpr, corte / max(otsu, 1)))

    linhas.sort(reverse=True)
    print(f"\nMedido em {n_img} imagens.\n")
    print(f"{'indice':<15}{'AUC':>7}{'melhor corte':>14}{'pega planta':>13}"
          f"{'pega fundo':>12}{'~thresh-scale':>15}")
    print("-" * 76)
    for auc, k, corte, tpr, fpr, escala in linhas:
        print(f"{k:<15}{auc:>7.3f}{corte:>14d}{tpr * 100:>12.0f}%{fpr * 100:>11.0f}%"
              f"{escala:>15.2f}")

    melhor_auc, melhor_k, corte, tpr, fpr, escala = linhas[0]
    print(f"\n== Veredito ==")
    if melhor_auc < 0.80:
        print(f"  O melhor indice ({melhor_k}) separa com AUC {melhor_auc:.3f} — fraco.")
        print("  Nenhum ajuste de limiar, tamanho ou forma recupera isso: a planta e")
        print("  o fundo tem praticamente a mesma cor nestas imagens. Abandone o")
        print("  estagio 0 e va direto para o YOLO, que aprende textura e formato,")
        print("  nao so cor.")
    else:
        print(f"  Use --index {melhor_k} (AUC {melhor_auc:.3f}): pega {tpr * 100:.0f}% "
              f"da planta deixando {fpr * 100:.0f}% do fundo passar.")
        print(f"    python src/cv_proposals.py --images ... --out ... \\")
        print(f"        --index {melhor_k} --thresh {corte} --plant-px <o seu>")
        print(f"  (ou, se preferir limiar relativo, --thresh-scale {escala:.2f})")
        auc_exg = [a for a, k, *_ in linhas if k == "exg"][0]
        if melhor_k != "exg" and melhor_auc - auc_exg > 0.03:
            print(f"\n  O exg (padrao) ficou em {auc_exg:.3f}. A troca de indice vale.")
        elif melhor_k != "exg":
            print(f"\n  Cuidado: o exg ficou em {auc_exg:.3f}, praticamente empatado.")
            print("  Trocar de indice NAO vai resolver seu problema — se o recall esta")
            print("  baixo com AUC alta, a cor esta separando bem e a falha e depois:")
            print("  morfologia (a roseta nao vira blob solido) ou separacao das")
            print("  plantas. Olhe o quadro 3 do --debug.")


if __name__ == "__main__":
    main()
