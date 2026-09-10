"""Varre um parametro do cv_proposals e MEDE cada valor contra os seus labels.

Existe porque o aviso de "vegetacao ocupa X%" e so uma heuristica — ela nao sabe
qual e a densidade do SEU plantio. Se voce tem imagens rotuladas, nao precisa
adivinhar: mede.

    python tools/varrer_parametros.py --param thresh_scale --valores 0.4 0.5 0.6 0.7
    python tools/varrer_parametros.py --param plant_px --valores auto 250 316 400
    python tools/varrer_parametros.py --param peak_sep --valores 0.35 0.45 0.6

Roda em memoria, sem escrever .txt nenhum. Cada valor leva o mesmo tempo de um
cv_proposals completo, entao comece com 3 ou 4 valores.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cv_proposals import detect                      # noqa: E402
from eval_proposals import match                     # noqa: E402
from tiling import label_path_for, list_images, read_yolo_labels  # noqa: E402

PADROES = dict(index="exg", thresh=None, thresh_scale=0.40, plant_px=None,
               min_area=0.20, max_area=3.0, min_solidity=0.55, max_aspect=2.5,
               peak_sep=0.45, max_radius=0.7, classe=0, max_lado=2000,
               abertura=0.04, fechamento=0.22)


def converter(nome: str, valor: str):
    if valor.lower() in ("auto", "none", ""):
        return None
    if nome in ("index",):
        return valor
    if nome in ("thresh", "classe", "max_lado"):
        return int(valor)
    return float(valor)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/raw/images"))
    ap.add_argument("--labels", type=Path, default=Path("data/raw/labels"))
    ap.add_argument("--param", required=True, choices=sorted(PADROES),
                    help="qual parametro variar")
    ap.add_argument("--valores", nargs="+", required=True,
                    help="valores a testar ('auto' = deixar o script estimar)")
    ap.add_argument("--iou", type=float, default=0.5)
    # os demais parametros ficam fixos no valor que voce passar aqui
    for nome, padrao in PADROES.items():
        ap.add_argument(f"--{nome.replace('_', '-')}", default=None,
                        help=f"fixa {nome} (padrao {padrao})")
    args = ap.parse_args()

    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit(f"nenhuma imagem em {args.images}")

    base = dict(PADROES)
    for nome in PADROES:
        v = getattr(args, nome, None)
        if v is not None and nome != args.param:
            base[nome] = converter(nome, str(v))

    # carrega imagens e labels uma vez so
    print(f"lendo {len(imgs)} imagens...")
    dados = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(label_path_for(p, args.labels), w, h)
        dados.append((p.stem, img, gt))
    total_gt = sum(len(g) for _, _, g in dados)

    print(f"\n{total_gt} agaves rotulados. Variando {args.param}:\n")
    print(f"{args.param:>14}{'propostas':>11}{'recall':>9}{'precisao':>10}"
          f"{'F1':>8}{'IoU':>7}{'cobert.':>9}{'plant_px':>10}")
    print("-" * 78)

    melhor = None
    for valor_txt in args.valores:
        cfg = dict(base)
        cfg[args.param] = converter(args.param, valor_txt)
        ns = types.SimpleNamespace(**cfg)

        TP = FP = FN = 0
        ious, cobs, ppxs, n_pred = [], [], [], 0
        for i, (_, img, gt) in enumerate(dados, 1):
            # progresso no stderr, para a tabela do stdout ficar limpa
            print(f"\r  {args.param}={valor_txt}: imagem {i}/{len(dados)}...",
                  end="", file=sys.stderr, flush=True)
            pred, info = detect(img, ns)
            cobs.append(info["cobertura"])
            ppxs.append(info["plant_px"])
            n_pred += len(pred)
            gt_ok, pred_ok = match(gt, pred, args.iou)
            TP += int(gt_ok.sum())
            FN += int((~gt_ok).sum())
            FP += int((~pred_ok).sum())
            if len(gt) and len(pred):
                from eval_proposals import iou_matrix
                ious.extend(iou_matrix(gt, pred).max(axis=1)[gt_ok].tolist())

        rec = TP / (TP + FN) if TP + FN else 0.0
        prec = TP / (TP + FP) if TP + FP else 0.0
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        print(" " * 60, end="\r", file=sys.stderr)     # limpa a linha de progresso
        print(f"{valor_txt:>14}{n_pred:>11}{rec:>9.3f}{prec:>10.3f}{f1:>8.3f}"
              f"{np.mean(ious) if ious else 0:>7.2f}"
              f"{np.mean(cobs) * 100:>8.1f}%{np.mean(ppxs):>10.0f}")
        if melhor is None or f1 > melhor[1]:
            melhor = (valor_txt, f1, rec, prec)

    if melhor:
        print(f"\nMelhor F1: --{args.param.replace('_', '-')} {melhor[0]}  "
              f"(recall {melhor[2]:.3f}, precisao {melhor[3]:.3f})")
        print("Para PRE-ANOTAR, prefira recall alto mesmo perdendo precisao: apagar\n"
              "caixa sobrando e rapido, desenhar a que faltou e lento.")


if __name__ == "__main__":
    main()
