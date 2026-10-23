"""Avalia o classificador de saude na classe que importa.

A acuracia (top1) que o Ultralytics reporta nao serve aqui. Com 93% de sadias
na validacao, um modelo que responde 'sadia' sempre tira 0,93 e nao achou uma
planta doente sequer.

O que importa e o RECALL na classe doente: de todas as doentes, quantas ele
acha. E, logo depois, a que custo em falso alarme.

    python tools/avaliar_cls.py --pesos runs/classify/agave_saude_v1/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=Path("data/crops/val"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    modelo = YOLO(str(args.pesos))
    nomes = modelo.names
    idx_doente = next(i for i, n in nomes.items() if n == "doente")

    arquivos, verdade = [], []
    for classe in ("doente", "sadia"):
        for p in sorted((args.val / classe).iterdir()):
            if p.suffix.lower() in EXT:
                arquivos.append(p)
                verdade.append(1 if classe == "doente" else 0)
    verdade = np.array(verdade)
    print(f"validacao: {len(arquivos)} recortes "
          f"({verdade.sum()} doentes, {(~verdade.astype(bool)).sum()} sadias)")

    probs = []
    for i in range(0, len(arquivos), 64):
        lote = [str(p) for p in arquivos[i:i + 64]]
        for r in modelo.predict(lote, device=args.device, verbose=False):
            probs.append(float(r.probs.data[idx_doente]))
    probs = np.array(probs)

    def metricas(t: float) -> tuple[int, int, int, float, float]:
        pred = probs >= t
        tp = int((pred & (verdade == 1)).sum())
        fp = int((pred & (verdade == 0)).sum())
        fn = int((~pred & (verdade == 1)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        return tp, fp, fn, rec, prec

    print(f"\n== Classe DOENTE, por limiar ==")
    print(f"{'limiar':>7} {'acha':>6} {'perde':>6} {'falso alarme':>13} "
          f"{'recall':>7} {'precisao':>9} {'F1':>6}")
    melhor = None
    for t in (0.10, 0.20, 0.30, 0.50, 0.70, 0.90):
        tp, fp, fn, rec, prec = metricas(t)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, t)
        print(f"{t:>7.2f} {tp:>6} {fn:>6} {fp:>13} {rec:>7.3f} {prec:>9.3f} {f1:>6.3f}")

    # AUC: mede a ordenacao, independente de limiar. E o numero certo para
    # decidir se o modelo serve como GARIMPEIRO (ranquear candidatos) mesmo
    # quando ainda nao serve como decisor.
    a = probs[verdade == 1][:, None]
    b = probs[verdade == 0][None, :]
    auc = float((a > b).mean() + 0.5 * (a == b).mean())
    print(f"\nAUC: {auc:.3f}   (0.5 = chute | score de cor atual = 0.889)")

    ordem = np.argsort(-probs)
    yo = verdade[ordem]
    n = len(yo)
    print(f"\n== Como garimpeiro: revisando de cima para baixo ==")
    for frac in (0.02, 0.05, 0.10, 0.20):
        k = max(1, int(n * frac))
        print(f"  top {frac*100:>4.0f}% ({k:>4} recortes) -> {yo[:k].sum():>3}/"
              f"{verdade.sum()} doentes")

    print(f"\n== Leitura ==")
    base = 1 - verdade.mean()
    print(f"  responder 'sadia' sempre daria acuracia {base:.3f} — e por isso")
    print(f"  que a acuracia nao serve para julgar este modelo.")
    if auc >= 0.90:
        print(f"  {auc:.3f} de AUC supera o score de cor (0.889): ja vale usar")
        print(f"  o modelo para garimpar as fotos novas.")
    elif auc >= 0.80:
        print(f"  {auc:.3f} de AUC: aprendeu sinal real, mas ainda nao supera o")
        print(f"  score de cor (0.889). Mais exemplos devem resolver.")
    else:
        print(f"  {auc:.3f} de AUC: pouco sinal. Com {verdade.sum()} doentes na")
        print(f"  validacao, isso aponta para falta de exemplo, nao de ajuste.")


if __name__ == "__main__":
    main()
