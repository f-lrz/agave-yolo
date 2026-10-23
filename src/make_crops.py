"""ETAPA 2 — extrai um recorte por agave e ordena por suspeita de doenca.

O grande ganho de reaproveitar a etapa 1: voce NAO desenha caixa nenhuma de
novo. Cada caixa rotulada vira um JPG de 224x224, e classificar vira arrastar
arquivo entre pastas no Explorer do Windows.

    python src/make_crops.py --images data/raw/images --labels data/raw/labels \
        --out data/crops

A TRIAGEM E INVERTIDA, de proposito:

    data/crops/_classificar/   todos os recortes ainda nao julgados
    data/crops/doente/         arraste PARA CA so as doentes

Com ~14.500 recortes e menos de 1% de doentes, mandar voce arrastar 14.432
arquivos para 'sadia' seria absurdo. Voce move apenas a classe rara; o que
sobrar em _classificar/ no fim e sadia por omissao. O tools/fechar_crops.py
monta o dataset de treino a partir disso.

ORDENACAO POR SUSPEITA: o nome do arquivo comeca com a posicao no ranking
(00001_, 00002_, ...), entao o Explorer ja mostra os mais suspeitos primeiro.
O score vem da cor do MIOLO do recorte (ver score_suspeita), medido contra as
68 caixas doentes que ja existiam nos labels:

    AUC 0.858
    top  5% dos recortes ( 728) contem 36/68 das doentes (53%)
    top 10% dos recortes (1456) contem 46/68 (68%)
    top 20% dos recortes (2913) contem 56/68 (82%)

Revise de cima para baixo: 20% do trabalho cobre ~80% das doentes. Mas NAO
pare so porque a taxa caiu — as ultimas ~12 estao espalhadas pelo resto, e
sao as que a cor nao enxerga (planta que adoeceu sem perder cor ainda).

Aviso honesto sobre esses numeros: eles foram medidos contra as suas 68
marcacoes, que voce fez de olho enquanto corrigia caixas. Se voce tendeu a
notar as plantas mais obviamente amarelas, o ranking parece melhor do que
sera para achar as doentes discretas.

As caixas ja marcadas como doente nos labels vao direto para doente/, sem
passar pela triagem.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tiling import label_path_for, list_images, read_yolo_labels


def score_suspeita(crop: np.ndarray, fracao_centro: float = 0.25) -> float:
    """Quanto maior, mais suspeito de estar doente.

    Eixo a* do Lab (verde <-> vermelho): planta amarelada/acinzentada desloca
    para o lado vermelho.

    SO O MIOLO CENTRAL ENTRA NA CONTA, e isso e o que faz a metrica funcionar.
    O recorte inclui a margem do --pad, que e quase toda solo avermelhado — e
    esse solo domina o a* de TODOS os recortes, afogando a diferenca entre as
    plantas. Medido nas 68 caixas doentes reais contra as 14.500 sadias:

        miolo    AUC    top 5%   top 10%   top 20%
          100%  0.553    5/68      9/68     17/68      <- recorte inteiro
           77%  0.609    9/68     12/68     21/68
           45%  0.805   20/68     29/68     43/68
           25%  0.858   36/68     46/68     56/68      <- padrao
           12%  0.835   38/68     48/68     50/68

    O platoe entre 0.20 e 0.30; a diferenca dentro dele e ruido para 68
    amostras. Abaixo de 0.20 sobram poucos pixels e fica instavel.
    """
    pequeno = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    k = max(4, int(round(64 * fracao_centro)))
    m = (64 - k) // 2
    miolo = pequeno[m:m + k, m:m + k]
    lab = cv2.cvtColor(miolo, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 1].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="margem extra ao redor da caixa; um pouco de solo e "
                         "vizinhanca ajuda o classificador a julgar o vigor")
    ap.add_argument("--min-px", type=int, default=24,
                    help="descarta recortes menores que isso (irreconheciveis)")
    ap.add_argument("--classe-doente", type=int, default=0,
                    help="indice da classe doente nos labels (ver classes.txt)")
    ap.add_argument("--score-centro", type=float, default=0.25,
                    help="fracao central do recorte usada no score de suspeita; "
                         "incluir a margem do --pad arruina a metrica (ver docstring)")
    ap.add_argument("--sem-ranking", action="store_true",
                    help="nao prefixa o nome com a posicao no ranking")
    ap.add_argument("--ranking-global", action="store_true",
                    help="ordena pelo score absoluto em vez do percentil dentro "
                         "de cada foto; so faz sentido se todas as fotos vierem "
                         "do mesmo voo e da mesma hora")
    ap.add_argument("--cls", type=Path, default=None,
                    help="pesos de um classificador yolov8-cls treinado. Se dado, "
                         "o score de suspeita vem da PROBABILIDADE de 'doente' do "
                         "modelo, nao da cor. O classificador treinado supera o "
                         "score de cor (AUC 0.934 vs 0.889), entao a triagem acha "
                         "as doentes mais rapido. Costuma ir melhor com "
                         "--ranking-global, por ser robusto a exposicao.")
    ap.add_argument("--cls-device", default="0",
                    help="device do classificador (--cls): '0' para GPU, 'cpu'")
    args = ap.parse_args()

    dest_tri = args.out / "_classificar"
    dest_doe = args.out / "doente"
    for p in (dest_tri, dest_doe):
        p.mkdir(parents=True, exist_ok=True)

    # Primeira passada: recorta tudo na memoria de trabalho minima possivel —
    # guardamos o recorte em JPG temporario na pasta certa so no fim, porque o
    # nome depende do ranking, que so existe depois de ver todos.
    registros: list[dict] = []
    descartados = 0

    for p in tqdm(list_images(args.images), desc="recortes"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        caixas = read_yolo_labels(label_path_for(p, args.labels), w, h)
        for i, (cl, x1, y1, x2, y2) in enumerate(caixas):
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < args.min_px:
                descartados += 1
                continue
            mx, my = bw * args.pad, bh * args.pad
            cx1, cy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
            cx2, cy2 = min(int(x2 + mx), w), min(int(y2 + my), h)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                descartados += 1
                continue
            crop = cv2.resize(crop, (args.size, args.size),
                              interpolation=cv2.INTER_AREA)
            registros.append({
                "imagem": p.name,
                "stem": p.stem,
                "indice": i,
                "classe_label": int(cl),
                "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                "score": score_suspeita(crop, args.score_centro),
                "_crop": crop,
            })

    if not registros:
        raise SystemExit(f"nenhuma caixa utilizavel em {args.labels}")

    # Se um classificador treinado foi passado, o score de suspeita deixa de ser
    # cor e passa a ser a probabilidade de 'doente' do modelo — que mede sinal
    # real, nao so amarelado (a planta que adoeceu sem perder cor, que a cor nao
    # ve, o classificador pega). Medido na validacao: AUC 0.934 vs 0.889 da cor.
    if args.cls is not None:
        from ultralytics import YOLO  # import tardio: estagio 0 nao usa torch
        print(f"ranqueando pelo classificador {args.cls.name} "
              f"(device {args.cls_device})...")
        modelo = YOLO(str(args.cls))
        idx_doente = next(i for i, n in modelo.names.items() if n == "doente")
        crops = [r["_crop"] for r in registros]
        for i in range(0, len(crops), 64):
            saida = modelo.predict(crops[i:i + 64], device=args.cls_device,
                                   imgsz=args.size, verbose=False)
            for j, res in enumerate(saida):
                registros[i + j]["score"] = float(res.probs.data[idx_doente])

    # O score bruto e um valor ABSOLUTO de cor, e fotos de voos diferentes tem
    # exposicao e angulo de sol diferentes. Comparar o a* de uma foto de meio-dia
    # com o de uma foto de fim de tarde mede a hora do voo, nao a planta.
    # Converter para percentil DENTRO da propria foto resolve isso:
    #
    #     score absoluto        AUC 0.859   top 5%: 36/68
    #     percentil na foto     AUC 0.889   top 5%: 39/68
    #
    # e 74% das doentes caem no top 10% da sua propria foto.
    if not args.ranking_global:
        por_foto: dict[str, list[dict]] = {}
        for r in registros:
            por_foto.setdefault(r["imagem"], []).append(r)
        for itens in por_foto.values():
            itens.sort(key=lambda r: r["score"])
            ultimo = max(len(itens) - 1, 1)
            for posicao, r in enumerate(itens):
                r["percentil"] = posicao / ultimo
    else:
        for r in registros:
            r["percentil"] = r["score"]

    # Ranking: mais suspeito primeiro. Empate desempata pelo score absoluto.
    registros.sort(key=lambda r: (-r["percentil"], -r["score"]))
    largura = len(str(len(registros)))

    n_doe = n_tri = 0
    linhas = [("recorte", "pasta", "imagem_origem", "indice_caixa",
               "classe_label", "score", "percentil_na_foto",
               "x1", "y1", "x2", "y2")]

    for pos, r in enumerate(registros, 1):
        prefixo = "" if args.sem_ranking else f"{pos:0{largura}d}_"
        nome = f"{prefixo}{r['stem']}__{r['indice']:04d}.jpg"
        # Caixa ja marcada como doente nos labels nao passa pela triagem.
        ja_doente = r["classe_label"] == args.classe_doente
        pasta = dest_doe if ja_doente else dest_tri
        cv2.imwrite(str(pasta / nome), r["_crop"],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ja_doente:
            n_doe += 1
        else:
            n_tri += 1
        linhas.append((nome, pasta.name, r["imagem"], r["indice"],
                       r["classe_label"], round(r["score"], 2),
                       round(r["percentil"], 4),
                       r["x1"], r["y1"], r["x2"], r["y2"]))

    idx = args.out / "crops_index.csv"
    with idx.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f, delimiter=";").writerows(linhas)

    print(f"\n{n_tri + n_doe} recortes  ({descartados} descartados por serem pequenos)")
    print(f"  {dest_doe}  ->  {n_doe} (ja marcadas como doente nos labels)")
    print(f"  {dest_tri}  ->  {n_tri} a triar, ordenados por suspeita")
    print(f"  {idx}  ->  mapa recorte -> caixa original\n")
    print("Agora, no Explorer:")
    print(f"  1. abra {dest_tri} com icones extra grandes")
    print("  2. os nomes comecam com a posicao no ranking, entao ordene por NOME")
    print(f"  3. arraste as doentes que encontrar para {dest_doe}")
    print("  4. o que sobrar aqui e sadia — nao precisa mover nada")
    print("\nQuando terminar:  python tools/fechar_crops.py --crops "
          f"{args.out}")
    if not args.sem_ranking:
        print("\nLembrete: a ordem acha ~metade das doentes nos primeiros 10%,")
        print("mas o resto esta espalhado. Nao pare so porque a taxa caiu.")


if __name__ == "__main__":
    main()
