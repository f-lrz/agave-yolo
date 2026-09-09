"""Mede propostas contra labels de verdade — precisao, recall e onde esta errando.

Aqui e onde as suas 9 imagens rotuladas rendem mais: como CONJUNTO DE MEDIDA,
nao de treino. Sem isso voce ajusta parametro no olho, olhando preview e
achando que melhorou.

Para gerar propostas, esta ferramenta serve tanto para o detector classico
quanto para o YOLO (predict_tiled.py --save-labels).

    python src/eval_proposals.py --pred data/preanot/labels \\
        --gt data/raw/labels --images data/raw/images

    # ver os erros desenhados: verde=acerto, vermelho=faltou, azul=falso positivo
    python src/eval_proposals.py --pred ... --gt ... --images ... --out out/erros
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tiling import list_images, read_yolo_labels

VERDE, VERMELHO, AZUL = (0, 200, 0), (0, 0, 255), (255, 150, 0)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de cada caixa de `a` contra cada de `b` -> matriz (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 1], b[None, :, 1])
    iy1 = np.maximum(a[:, None, 2], b[None, :, 2])
    ix2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iy2 = np.minimum(a[:, None, 4], b[None, :, 4])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 3] - a[:, 1]) * (a[:, 4] - a[:, 2])
    bb = (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Casamento guloso 1-para-1 por IoU decrescente.

    Guloso e nao hungaro de proposito: e o mesmo criterio que a metrica COCO usa,
    e evita que uma predicao gigante 'cubra' varios GT.
    """
    M = iou_matrix(gt, pred)
    gt_ok = np.zeros(len(gt), bool)
    pred_ok = np.zeros(len(pred), bool)
    if M.size:
        for gi, pi in sorted(np.ndindex(M.shape), key=lambda ij: -M[ij]):
            if M[gi, pi] < thr:
                break
            if not gt_ok[gi] and not pred_ok[pi]:
                gt_ok[gi] = pred_ok[pi] = True
    return gt_ok, pred_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True, help="pasta com os .txt previstos")
    ap.add_argument("--gt", type=Path, required=True, help="pasta com os .txt de verdade")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None, help="salvar imagens dos erros")
    args = ap.parse_args()

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    TP = FP = FN = 0
    linhas = []
    ious_ok: list[float] = []

    for p in list_images(args.images):
        gt_file = args.gt / f"{p.stem}.txt"
        if not gt_file.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(gt_file, w, h)
        pred = read_yolo_labels(args.pred / f"{p.stem}.txt", w, h)
        gt_ok, pred_ok = match(gt, pred, args.iou)

        tp, fn, fp = int(gt_ok.sum()), int((~gt_ok).sum()), int((~pred_ok).sum())
        TP, FN, FP = TP + tp, FN + fn, FP + fp
        rec = tp / len(gt) if len(gt) else 0.0
        prec = tp / len(pred) if len(pred) else 0.0
        linhas.append((p.stem, len(gt), len(pred), tp, fn, fp, rec, prec))

        if len(gt) and len(pred):
            M = iou_matrix(gt, pred)
            ious_ok.extend(M.max(axis=1)[gt_ok].tolist())

        if args.out:
            vis = img.copy()
            t = max(2, round(min(h, w) / 700))
            for b, ok in zip(pred, pred_ok):
                if not ok:
                    cv2.rectangle(vis, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), AZUL, t)
            for b, ok in zip(gt, gt_ok):
                cv2.rectangle(vis, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])),
                              VERDE if ok else VERMELHO, t)
            cv2.imwrite(str(args.out / f"{p.stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])

    if not linhas:
        raise SystemExit("nenhuma imagem com label de verdade encontrada")

    print(f"\n{'imagem':<24}{'gt':>5}{'pred':>6}{'TP':>5}{'FN':>5}{'FP':>5}"
          f"{'recall':>9}{'precis':>9}")
    for nome, ngt, npr, tp, fn, fp, rec, prec in linhas:
        print(f"{nome:<24}{ngt:>5}{npr:>6}{tp:>5}{fn:>5}{fp:>5}{rec:>9.3f}{prec:>9.3f}")

    rec = TP / (TP + FN) if TP + FN else 0.0
    prec = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    print(f"\n== Total (IoU >= {args.iou}) ==")
    print(f"  recall    {rec:.3f}   ({TP} de {TP + FN} agaves reais encontrados)")
    print(f"  precisao  {prec:.3f}   ({FP} caixas sobrando para apagar)")
    print(f"  F1        {f1:.3f}")
    if ious_ok:
        print(f"  IoU medio dos acertos: {np.mean(ious_ok):.3f}  "
              f"(baixo = caixa encostando mal na planta)")

    print("\n== Leitura ==")
    if rec < 0.6:
        print("  Recall baixo: o problema esta na SEGMENTACAO, nao nos filtros.")
        print("  Rode com --debug e olhe o painel 3 (mascara): a planta esta la?")
        print("  Se nao estiver, troque --index (exgr, vari) ou baixe --thresh.")
    elif prec < 0.5:
        print("  Muito falso positivo: suba --min-solidity e --min-area.")
        print("  Ainda assim, para PRE-ANOTAR isso pode ser aceitavel — apagar")
        print("  caixa e rapido, desenhar a que faltou e lento.")
    elif rec > 0.75:
        print("  Bom o suficiente para pre-anotar. Corrija estas propostas,")
        print("  treine o YOLO semente, e a partir da o proprio YOLO assume.")
    if ious_ok and np.mean(ious_ok) < 0.65:
        print("  IoU dos acertos baixo: as caixas estao encontrando a planta mas")
        print("  mal ajustadas. Ajuste o fechamento morfologico (--plant-px).")


if __name__ == "__main__":
    main()
