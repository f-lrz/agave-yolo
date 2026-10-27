"""Teste manual do pipeline de saude: foto -> detecta agaves -> marca as DOENTES.

    python teste_saude/testar.py

Le tudo de teste_saude/entrada/ e escreve em teste_saude/saida/:

    saida/imagens/<nome>_doentes.jpg   a foto com SO as caixas de doente
    saida/json/<nome>.json             laudo completo (todas as plantas)
    saida/resumo.csv                   uma linha por foto

A foto original em entrada/ NUNCA e tocada — o script le os bytes e decodifica
em memoria, nunca abre o arquivo para escrita.

Usa o mesmo AgavePipeline de src/pipeline.py (detector + classificador), para o
que voce ve aqui ser o que o backend vai produzir.

Opcoes:
    --limiar 0.5     prob_doente a partir da qual a planta e marcada (desenho).
                     Abaixe (ex. 0.3) para ver mais suspeitas; suba para so as
                     mais certas. NAO reclassifica o laudo, so o que e desenhado.
    --conf 0.30      limiar de DETECCAO (calibrado para contagem; detector)
    --device cpu     '0' para GPU (esta maquina usa cpu)
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
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))

from pipeline import AgavePipeline  # noqa: E402

VERMELHO = (0, 0, 255)  # BGR
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
       ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP"}


def desenhar_doentes(img: np.ndarray, laudo: dict, limiar: float) -> tuple[np.ndarray, int]:
    """Desenha SO as plantas com prob_doente >= limiar. Devolve (imagem, n)."""
    out = img.copy()
    thick = max(2, round(min(img.shape[:2]) / 700))
    n = 0
    for p in laudo["plantas"]:
        prob = p.get("prob_doente")
        if prob is None or prob < limiar:
            continue
        n += 1
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), VERMELHO, thick)
        cv2.putText(out, f"{prob:.2f}", (x1, max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick, VERMELHO, thick, cv2.LINE_AA)
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos-det", type=Path, default=RAIZ / "best_v5.pt")
    ap.add_argument("--pesos-cls", type=Path,
                    default=RAIZ / "runs" / "classify" / "agave_saude_v3_final" / "weights" / "best.pt")
    ap.add_argument("--entrada", type=Path, default=AQUI / "entrada")
    ap.add_argument("--saida", type=Path, default=AQUI / "saida")
    ap.add_argument("--limiar", type=float, default=0.5,
                    help="prob_doente minima para marcar a planta (so no desenho)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="limiar de deteccao (0.30 = calibrado para contagem)")
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
    d_jsn = args.saida / "json"
    for p in (d_img, d_jsn):
        p.mkdir(parents=True, exist_ok=True)

    print(f"detector      : {args.pesos_det}")
    print(f"classificador : {args.pesos_cls}")
    print(f"conf deteccao : {args.conf}   limiar doente (desenho): {args.limiar}")
    pipe = AgavePipeline(str(args.pesos_det), str(args.pesos_cls),
                         conf=args.conf, device=args.device)

    linhas_csv = []
    for i, foto in enumerate(fotos, 1):
        nome = foto.stem
        destino_img = d_img / f"{nome}_doentes.jpg"
        if destino_img.exists() and not args.refazer:
            print(f"[{i}/{len(fotos)}] {foto.name}: ja processada (use --refazer)")
            continue

        print(f"[{i}/{len(fotos)}] {foto.name} ... ", end="", flush=True)
        # le os bytes, igual ao backend: nao abre o arquivo para escrita
        img = cv2.imdecode(np.frombuffer(foto.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("ERRO: nao consegui decodificar a imagem")
            linhas_csv.append([foto.name, "", "", "", "", "", "", "", "ERRO: decodificacao"])
            continue

        h, w = img.shape[:2]
        t0 = time.perf_counter()
        laudo = pipe.run(img)
        dt = time.perf_counter() - t0

        anotada, n_marcadas = desenhar_doentes(img, laudo, args.limiar)
        cv2.imwrite(str(destino_img), anotada, [cv2.IMWRITE_JPEG_QUALITY, args.qualidade])
        (d_jsn / f"{nome}.json").write_text(
            json.dumps({"arquivo": foto.name, **laudo}, ensure_ascii=False, indent=1),
            encoding="utf-8")

        total = laudo["total_agaves"]
        print(f"{total} agaves, {n_marcadas} doentes marcadas (>= {args.limiar}) em {dt:.1f}s")
        linhas_csv.append([foto.name, w, h, total, laudo["doentes"],
                           laudo["sadias"], laudo["taxa_doenca"], args.limiar, round(dt, 1)])

    if linhas_csv:
        csv_path = args.saida / "resumo.csv"
        novo = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if novo:
                w.writerow(["arquivo", "largura", "altura", "total_agaves",
                            "doentes_laudo", "sadias", "taxa_doenca", "limiar_desenho",
                            "segundos"])
            w.writerows(linhas_csv)
        print(f"\nresumo -> {csv_path}")

    print(f"imagens (so doentes) -> {d_img}")
    print(f"laudo json completo  -> {d_jsn}")
    print(f"originais intactos   -> {args.entrada}")


if __name__ == "__main__":
    main()
