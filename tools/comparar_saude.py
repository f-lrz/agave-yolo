"""Compara dois laudos do teste_saude (ex.: v4-final x v5-final) planta a planta.

    python tools/comparar_saude.py --a teste_saude/saida_v4 --b teste_saude/saida_v5 \
        --entrada data/novas_voo3/teste --out teste_saude/comparacao

Os dois laudos usam o MESMO detector, entao as plantas saem na mesma ordem e com
as mesmas caixas; so muda o prob_doente. O script:

  - imprime uma tabela por foto: doentes de A, doentes de B, e quantas divergem;
  - monta folhas com os recortes das DIVERGENCIAS:
        <out>/so_A.jpg   plantas que SO o modelo A marcou como doente
        <out>/so_B.jpg   plantas que SO o modelo B marcou como doente
    cada recorte com "A 0.93 | B 0.12" (prob_doente de cada modelo).

Olhar so as divergencias e o jeito rapido de julgar: onde os dois concordam,
nao ha nada para decidir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def recorte(img: np.ndarray, bbox, pad: float = 0.15, size: int = 200) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * pad, (y2 - y1) * pad
    c = img[max(int(y1 - my), 0):min(int(y2 + my), h),
            max(int(x1 - mx), 0):min(int(x2 + mx), w)]
    if c.size == 0:
        c = np.zeros((size, size, 3), np.uint8)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def folha(itens: list[tuple[np.ndarray, str]], destino: Path, cols: int = 8) -> None:
    if not itens:
        return
    cell, rot, m = 200, 34, 6
    rows = (len(itens) + cols - 1) // cols
    canvas = np.full((rows * (cell + rot + m) + m, cols * (cell + m) + m, 3), 245, np.uint8)
    for i, (img, texto) in enumerate(itens):
        r, c = divmod(i, cols)
        oy, ox = m + r * (cell + rot + m), m + c * (cell + m)
        canvas[oy + rot:oy + rot + cell, ox:ox + cell] = img
        for k, linha in enumerate(texto.split("\n")):
            cv2.putText(canvas, linha, (ox + 2, oy + 13 + k * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(destino), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="pasta saida do modelo A")
    ap.add_argument("--b", type=Path, required=True, help="pasta saida do modelo B")
    ap.add_argument("--entrada", type=Path, required=True, help="fotos originais")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limiar", type=float, default=0.5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    so_a: list[tuple[np.ndarray, str]] = []
    so_b: list[tuple[np.ndarray, str]] = []
    tot = {"agaves": 0, "a": 0, "b": 0, "ambos": 0, "so_a": 0, "so_b": 0}

    print(f"{'foto':14s} {'agaves':>6s} {'doen A':>7s} {'doen B':>7s} {'ambos':>6s} {'so A':>5s} {'so B':>5s}")
    for ja in sorted((args.a / "json").glob("*.json")):
        jb = args.b / "json" / ja.name
        if not jb.exists():
            print(f"{ja.stem}: sem laudo em B, pulando")
            continue
        la = json.loads(ja.read_text(encoding="utf-8"))
        lb = json.loads(jb.read_text(encoding="utf-8"))
        pa, pb = la["plantas"], lb["plantas"]
        if len(pa) != len(pb) or any(x["bbox"] != y["bbox"] for x, y in zip(pa, pb)):
            print(f"{ja.stem}: caixas diferentes entre A e B (detector/conf mudou?), pulando")
            continue

        img = None
        foto = args.entrada / la.get("arquivo", ja.stem + ".JPG")
        if foto.exists():
            img = cv2.imdecode(np.frombuffer(foto.read_bytes(), np.uint8), cv2.IMREAD_COLOR)

        n = {"a": 0, "b": 0, "ambos": 0, "so_a": 0, "so_b": 0}
        for i, (x, y) in enumerate(zip(pa, pb)):
            da = (x["prob_doente"] or 0) >= args.limiar
            db = (y["prob_doente"] or 0) >= args.limiar
            n["a"] += da
            n["b"] += db
            n["ambos"] += da and db
            if da != db:
                chave = "so_a" if da else "so_b"
                n[chave] += 1
                if img is not None:
                    txt = f"{ja.stem} #{i}\nA {x['prob_doente']:.2f} | B {y['prob_doente']:.2f}"
                    (so_a if da else so_b).append((recorte(img, x["bbox"]), txt))

        print(f"{ja.stem:14s} {len(pa):6d} {n['a']:7d} {n['b']:7d} {n['ambos']:6d} {n['so_a']:5d} {n['so_b']:5d}")
        tot["agaves"] += len(pa)
        for k in n:
            tot[k] += n[k]

    print(f"{'TOTAL':14s} {tot['agaves']:6d} {tot['a']:7d} {tot['b']:7d} {tot['ambos']:6d} {tot['so_a']:5d} {tot['so_b']:5d}")
    folha(so_a, args.out / "so_A.jpg")
    folha(so_b, args.out / "so_B.jpg")
    print(f"\nA = {args.a}\nB = {args.b}")
    print(f"divergencias: {args.out / 'so_A.jpg'} ({len(so_a)})  |  {args.out / 'so_B.jpg'} ({len(so_b)})")


if __name__ == "__main__":
    main()
