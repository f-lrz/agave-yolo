"""Teste manual do detector: joga foto em entrada/, roda, olha o resultado.

    python teste_manual/testar.py

Le tudo de teste_manual/entrada/ e escreve em teste_manual/saida/:

    saida/imagens/<nome>_caixas.jpg   a foto com as caixas desenhadas
    saida/labels/<nome>.txt           label YOLO normalizado
    saida/json/<nome>.json            saida completa, com pixels e confianca
    saida/resumo.csv                  uma linha por foto

A foto original em entrada/ NUNCA e tocada — nem lida em modo de escrita.

Usa o mesmo codigo da entrega (entrega_detector/agave_detector), para o que
voce ve aqui ser o que o backend vai produzir.

Opcoes:
    --conf 0.30      limiar. 0.30 = calibrado para contagem; 0.20 = pre-anotar
    --classe 1       indice escrito no .txt (1 = sadia, pela convencao do projeto)
    --cor preto      preto | magenta | ciano  (magenta ajuda em foto saturada)
    --device cpu     '0' para GPU
    --refazer        reprocessa fotos que ja tem saida
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "entrega_detector"))

from agave_detector import (CIANO, CONF_PADRAO, MAGENTA, PRETO,  # noqa: E402
                            DetectorAgave, desenhar_caixas, ler_imagem)

CORES = {"preto": PRETO, "magenta": MAGENTA, "ciano": CIANO}
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
       ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP"}


def para_yolo(caixas: list[dict], w: int, h: int, classe: int) -> list[str]:
    """[x1,y1,x2,y2] em pixels -> 'classe cx cy largura altura' normalizado."""
    linhas = []
    for b in caixas:
        cx = ((b["x1"] + b["x2"]) / 2) / w
        cy = ((b["y1"] + b["y2"]) / 2) / h
        bw = (b["x2"] - b["x1"]) / w
        bh = (b["y2"] - b["y1"]) / h
        if bw <= 0 or bh <= 0:
            continue
        linhas.append(f"{classe} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", type=Path, default=AQUI.parent / "entrega_detector" / "modelo" / "best.pt")
    ap.add_argument("--entrada", type=Path, default=AQUI / "entrada")
    ap.add_argument("--saida", type=Path, default=AQUI / "saida")
    ap.add_argument("--conf", type=float, default=CONF_PADRAO)
    ap.add_argument("--classe", type=int, default=1,
                    help="indice no .txt; 1 = sadia na convencao do projeto")
    ap.add_argument("--cor", choices=list(CORES), default="preto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--qualidade", type=int, default=88)
    ap.add_argument("--refazer", action="store_true")
    args = ap.parse_args()

    fotos = sorted(p for p in args.entrada.iterdir()
                   if p.is_file() and p.suffix in EXT) if args.entrada.exists() else []
    if not fotos:
        raise SystemExit(
            f"nenhuma imagem em {args.entrada}\n"
            f"Coloque as fotos la (o arquivo original fica intacto) e rode de novo."
        )

    d_img = args.saida / "imagens"
    d_lbl = args.saida / "labels"
    d_jsn = args.saida / "json"
    for p in (d_img, d_lbl, d_jsn):
        p.mkdir(parents=True, exist_ok=True)

    print(f"pesos : {args.pesos}")
    print(f"conf  : {args.conf}   classe no .txt: {args.classe}   cor: {args.cor}")
    det = DetectorAgave(args.pesos, device=args.device, conf=args.conf)

    linhas_csv = []
    for i, foto in enumerate(fotos, 1):
        nome = foto.stem
        destino_img = d_img / f"{nome}_caixas.jpg"
        if destino_img.exists() and not args.refazer:
            print(f"[{i}/{len(fotos)}] {foto.name}: ja processada (use --refazer)")
            continue

        print(f"[{i}/{len(fotos)}] {foto.name} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            # le os bytes, igual ao backend: nao abre o arquivo para escrita
            r = det.detectar(foto.read_bytes())
        except ValueError as e:
            print(f"ERRO: {e}")
            linhas_csv.append([foto.name, "", "", "", "", "", f"ERRO: {e}"])
            continue
        dt = time.perf_counter() - t0

        img = desenhar_caixas(ler_imagem(foto), r["caixas"], CORES[args.cor])
        cv2.imwrite(str(destino_img), img, [cv2.IMWRITE_JPEG_QUALITY, args.qualidade])
        (d_lbl / f"{nome}.txt").write_text(
            "\n".join(para_yolo(r["caixas"], r["largura"], r["altura"], args.classe)) + "\n",
            encoding="utf-8")
        (d_jsn / f"{nome}.json").write_text(
            json.dumps({"arquivo": foto.name, **r}, ensure_ascii=False, indent=1),
            encoding="utf-8")

        print(f"{r['total']} agaves em {dt:.1f}s")
        for aviso in r["avisos"]:
            print(f"      AVISO: {aviso}")
        linhas_csv.append([foto.name, r["largura"], r["altura"], r["total"],
                           args.conf, round(dt, 1), " | ".join(r["avisos"])])

    if linhas_csv:
        csv_path = args.saida / "resumo.csv"
        novo = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if novo:
                w.writerow(["arquivo", "largura", "altura", "total_agaves",
                            "conf", "segundos", "avisos"])
            w.writerows(linhas_csv)
        print(f"\nresumo -> {csv_path}")

    print(f"imagens com caixas -> {d_img}")
    print(f"labels YOLO        -> {d_lbl}")
    print(f"json completo      -> {d_jsn}")
    print(f"originais intactos -> {args.entrada}")


if __name__ == "__main__":
    main()
