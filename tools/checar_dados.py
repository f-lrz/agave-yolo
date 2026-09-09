"""Confere se as pastas de dados estao montadas certo, ANTES de rodar qualquer coisa.

Pega os erros silenciosos: label com nome que nao bate com a imagem, .txt em
formato errado (Pascal VOC / COCO / pixels absolutos em vez de YOLO normalizado),
imagem repetida entre as pastas.

    python tools/checar_dados.py
    python tools/checar_dados.py --raiz data/raw
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tiling import list_images  # noqa: E402

OK, AVISO, ERRO = "  ok ", "  !! ", "  XX "


def checar_label(path: Path, classes: Counter[int] | None = None) -> tuple[int, list[str]]:
    """-> (numero de caixas, problemas encontrados)"""
    problemas, n = [], 0
    for i, linha in enumerate(path.read_text().splitlines(), 1):
        partes = linha.split()
        if not partes:
            continue
        n += 1
        if classes is not None and partes[0].lstrip("-").isdigit():
            classes[int(partes[0])] += 1
        if len(partes) != 5:
            problemas.append(f"linha {i}: {len(partes)} colunas, esperado 5 "
                             f"(classe cx cy largura altura)")
            continue
        try:
            valores = [float(x) for x in partes]
        except ValueError:
            problemas.append(f"linha {i}: valor nao numerico")
            continue
        if any(v < 0 or v > 1 for v in valores[1:]):
            problemas.append(f"linha {i}: coordenadas fora de 0-1 — o arquivo "
                             f"parece estar em pixels absolutos, nao YOLO normalizado")
        if valores[3] <= 0 or valores[4] <= 0:
            problemas.append(f"linha {i}: largura ou altura zero")
    return n, problemas[:3]          # so os 3 primeiros, para nao poluir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    d_img, d_lab, d_nao = args.raiz / "images", args.raiz / "labels", args.raiz / "nao_rotuladas"
    falhou = False

    # classes.txt: um nome por linha, na ordem dos ids (linha 1 = classe 0).
    # E a convencao do LabelImg, e serve para a correspondencia id -> nome nao
    # ficar so na sua memoria — em dois meses ninguem lembra se 0 era doente.
    f_classes = args.raiz / "classes.txt"
    nomes: dict[int, str] = {}
    if f_classes.exists():
        nomes = {i: n.strip() for i, n in enumerate(f_classes.read_text().splitlines())
                 if n.strip()}

    print(f"\n== Conjunto de MEDIDA ({d_img}) ==")
    if not d_img.exists():
        print(ERRO + f"pasta nao existe: {d_img}")
        return
    imgs = list_images(d_img)
    if not imgs:
        print(ERRO + "nenhuma imagem aqui. Coloque as que voce rotulou a mao.")
        falhou = True

    total_caixas, sem_label = 0, []
    classes: Counter[int] = Counter()
    for p in imgs:
        lab = d_lab / f"{p.stem}.txt"
        if not lab.exists():
            sem_label.append(p.name)
            continue
        n, problemas = checar_label(lab, classes)
        total_caixas += n
        if problemas:
            falhou = True
            print(ERRO + f"{p.name}:")
            for pr in problemas:
                print(f"        {pr}")
        elif n == 0:
            print(AVISO + f"{p.name}: label existe mas esta vazio")

    if sem_label:
        falhou = True
        print(ERRO + f"{len(sem_label)} imagem(ns) sem .txt correspondente em {d_lab}:")
        for nome in sem_label[:5]:
            print(f"        {nome}  ->  falta {Path(nome).stem}.txt")
        print("        (o .txt precisa ter o MESMO nome-base da imagem)")

    orfaos = [f.name for f in sorted(d_lab.glob("*.txt"))
              if not any(p.stem == f.stem for p in imgs)] if d_lab.exists() else []
    if orfaos:
        print(AVISO + f"{len(orfaos)} .txt sem imagem correspondente: "
                      f"{', '.join(orfaos[:5])}")

    if imgs and not sem_label:
        med = total_caixas / len(imgs)
        print(OK + f"{len(imgs)} imagens, {total_caixas} caixas ({med:.0f} por imagem)")
        if len(classes) > 1:
            dist = ", ".join(f"{c}={nomes.get(c, '?')}: {n}" if nomes else f"classe {c}: {n}"
                             for c, n in sorted(classes.items()))
            print(OK + f"{len(classes)} classes -> {dist}")
            if not nomes:
                print(AVISO + f"crie {f_classes} com um nome por linha (linha 1 = "
                              f"classe 0) para registrar qual id e qual")
            print(AVISO + "Para o detector do estagio 1, colapse tudo em uma classe:")
            print(f"        python tools/juntar_classes.py --entrada {d_lab} "
                  f"--saida {d_lab.parent / (d_lab.name + '_1classe')}")
            raras = [c for c, n in classes.items() if n < 50]
            if raras:
                rot = ", ".join(f"{c} ({nomes.get(c, '?')})" if nomes else str(c)
                                for c in raras)
                print(AVISO + f"classe(s) {rot} com pouquissimos exemplos — "
                              f"insuficiente para treinar o estagio 2 por enquanto")
            maior = max(classes, key=classes.get)
            print(AVISO + f"ao pre-anotar, emita a classe majoritaria ({maior}"
                          f"{' = ' + nomes[maior] if maior in nomes else ''}): "
                          f"cv_proposals.py --classe {maior}")
        amostra = cv2.imread(str(imgs[0]))
        if amostra is not None:
            h, w = amostra.shape[:2]
            print(OK + f"resolucao: {w}x{h}")
            if max(w, h) < 1500:
                print(AVISO + "imagem pequena para foto de drone — confira se "
                              "nao redimensionaram o arquivo")

    print(f"\n== Imagens a pre-anotar ({d_nao}) ==")
    if not d_nao.exists():
        print(AVISO + f"pasta nao existe ainda: {d_nao}")
    else:
        nao = list_images(d_nao)
        print(OK + f"{len(nao)} imagens" if nao else AVISO + "vazia")
        repetidas = {p.stem for p in nao} & {p.stem for p in imgs}
        if repetidas:
            print(AVISO + f"{len(repetidas)} imagem(ns) estao NAS DUAS pastas: "
                          f"{', '.join(sorted(repetidas)[:5])}")
            print("        Isso contamina a medida. Deixe cada imagem em uma pasta so.")

    print("\n" + ("XX  Corrija os erros acima antes de seguir."
                  if falhou else "ok  Tudo certo. Pode seguir para o passo 1."))


if __name__ == "__main__":
    main()
