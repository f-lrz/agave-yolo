"""Folha(s) de contato de uma pasta de recortes, para revisar tudo de uma vez.

    python tools/montagem.py <pasta> <prefixo_saida> [por_folha] [cols]

Numera as celulas 1..N na ordem alfabetica dos arquivos — a MESMA ordem que o
Explorer do Windows mostra. Assim da para revisar de dois jeitos:

  - arrastar os recortes no Explorer para uma subpasta, ou
  - anotar os numeros ("secas: 5, 12, 40-44") e passar para o assistente.

Ignora subpastas (so le arquivos de imagem diretos da pasta).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    pasta = Path(sys.argv[1])
    prefixo = sys.argv[2]
    por_folha = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    cols = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    cell, margem, rotulo = 150, 6, 18

    arquivos = sorted(p for p in pasta.iterdir()
                      if p.is_file() and p.suffix.lower() in EXT)
    if not arquivos:
        raise SystemExit(f"nenhuma imagem em {pasta}")

    cw, ch = cell + margem, cell + rotulo + margem
    n_folhas = (len(arquivos) + por_folha - 1) // por_folha
    for f in range(n_folhas):
        bloco = arquivos[f * por_folha:(f + 1) * por_folha]
        rows = (len(bloco) + cols - 1) // cols
        canvas = np.full((rows * ch + margem, cols * cw + margem, 3), 245, np.uint8)
        for i, p in enumerate(bloco):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            oy, ox = margem + r * ch, margem + c * cw
            canvas[oy + rotulo:oy + rotulo + cell, ox:ox + cell] = img
            num = f * por_folha + i + 1
            cv2.putText(canvas, str(num), (ox + 2, oy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        saida = f"{prefixo}_folha{f + 1}.jpg"
        cv2.imwrite(saida, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ini, fim = f * por_folha + 1, f * por_folha + len(bloco)
        print(f"folha {f + 1}: #{ini}..{fim} ({len(bloco)} recortes) -> {saida}")


if __name__ == "__main__":
    main()
