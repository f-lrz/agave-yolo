"""Colapsa labels de varias classes em uma so, para o detector do estagio 1.

Nao apaga nada: le de uma pasta e escreve em outra. Os labels originais de 2
classes continuam intactos, e sao eles que alimentam o estagio 2 depois
(make_crops.py --names ... usa a classe para pre-separar os recortes).

Por que colapsar: o estagio 1 tem que achar TODO agave, doente ou nao. Se o
detector tambem tiver que decidir a classe, a classe rara vira caixa perdida —
a planta doente simplesmente nao e detectada, que e o unico erro que este
projeto nao pode cometer.

    python tools/juntar_classes.py --entrada data/raw/labels --saida data/raw/labels_1classe
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--classe", type=int, default=0, help="classe de destino")
    args = ap.parse_args()

    if args.saida.resolve() == args.entrada.resolve():
        raise SystemExit("saida nao pode ser igual a entrada — o objetivo e "
                         "preservar os labels originais")

    args.saida.mkdir(parents=True, exist_ok=True)
    contagem: Counter[str] = Counter()
    n_arquivos = 0

    for txt in sorted(args.entrada.glob("*.txt")):
        linhas = []
        for linha in txt.read_text().splitlines():
            partes = linha.split()
            if len(partes) < 5:
                continue
            contagem[partes[0]] += 1
            linhas.append(" ".join([str(args.classe), *partes[1:5]]))
        (args.saida / txt.name).write_text("\n".join(linhas) + "\n")
        n_arquivos += 1

    print(f"\n{n_arquivos} arquivos -> {args.saida}")
    print("\nclasses encontradas na entrada:")
    for cls, n in sorted(contagem.items()):
        print(f"  classe {cls}: {n} caixas")
    print(f"\ntodas viraram classe {args.classe}. "
          f"Os labels originais em {args.entrada} continuam intactos.")


if __name__ == "__main__":
    main()
