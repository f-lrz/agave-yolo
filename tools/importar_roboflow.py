"""Traz um export YOLOv8 do Roboflow de volta para data/raw/, traduzindo as classes.

O problema que este script resolve: o Roboflow so exporta as classes que
REALMENTE aparecem no lote, e reindexa a partir de 0. Se voce corrigiu um lote
onde nenhuma planta foi marcada como doente, o export volta com

    nc: 1
    names: ['sadia']

e todas as caixas com indice 0 — que na convencao do projeto (classes.txt) e
DOENTE. Copiar assim inverte o significado dos rotulos sem dar erro nenhum, e
o estrago so aparece no estagio 2, quando o classificador ja foi treinado.

Este script le os nomes do data.yaml do export, le a convencao de classes.txt,
e reescreve cada indice pelo NOME — nunca pela posicao.

    python tools/importar_roboflow.py --export ~/Downloads/meu_export.yolov8
    python tools/importar_roboflow.py --export ... --mover-originais   # tira de nao_rotuladas
    python tools/importar_roboflow.py --export ... --simular           # so mostra o que faria
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

OK, AVISO, ERRO = "  ok ", "  !! ", "  XX "


def nomes_do_export(data_yaml: Path) -> dict[int, str]:
    """names: pode vir como lista ['a','b'] ou dict {0: a, 1: b}."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names")
    if names is None:
        raise SystemExit(f"{ERRO}{data_yaml} nao tem 'names'")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def convencao_do_projeto(classes_txt: Path) -> dict[str, int]:
    """classes.txt: uma classe por linha, a ordem define o indice."""
    linhas = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines()]
    return {nome: i for i, nome in enumerate(linhas) if nome}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True,
                    help="pasta descompactada do export YOLOv8 do Roboflow")
    ap.add_argument("--raiz", type=Path, default=Path("data/raw"),
                    help="destino: <raiz>/images e <raiz>/labels")
    ap.add_argument("--mover-originais", action="store_true",
                    help="move as imagens correspondentes de nao_rotuladas/ para "
                         "ja_corrigidas/, para nao pre-anotar de novo")
    ap.add_argument("--simular", action="store_true",
                    help="mostra o que faria sem escrever nada")
    args = ap.parse_args()

    data_yaml = args.export / "data.yaml"
    classes_txt = args.raiz / "classes.txt"
    for p in (data_yaml, classes_txt):
        if not p.exists():
            raise SystemExit(f"{ERRO}nao encontrei {p}")

    export_nomes = nomes_do_export(data_yaml)
    projeto = convencao_do_projeto(classes_txt)

    print("\n== Traducao de classes ==")
    remap: dict[int, int] = {}
    for idx, nome in sorted(export_nomes.items()):
        if nome not in projeto:
            raise SystemExit(
                f"{ERRO}o export tem a classe '{nome}', que nao existe em "
                f"{classes_txt} ({', '.join(projeto)}).\n"
                f"     Corrija o nome no Roboflow ou adicione em classes.txt — "
                f"adivinhar aqui seria pior que parar.")
        remap[idx] = projeto[nome]
        seta = "->" if idx != projeto[nome] else "=="
        print(f"  export {idx} ({nome}) {seta} projeto {projeto[nome]}")
    if any(k != v for k, v in remap.items()):
        print(f"{AVISO}os indices MUDAM. Era exatamente este o risco.")
    else:
        print(f"{OK}os indices ja coincidem.")

    img_dest = args.raiz / "images"
    lbl_dest = args.raiz / "labels"
    if not args.simular:
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

    # o export vem em train/valid/test; o make_tiles faz o proprio split por
    # imagem de origem, entao achatamos tudo aqui.
    contagem: Counter[str] = Counter()
    n_img = n_lbl = n_caixas = 0
    sobrescritas: list[str] = []
    stems: list[str] = []

    for split in ("train", "valid", "test"):
        pasta_img = args.export / split / "images"
        pasta_lbl = args.export / split / "labels"
        if not pasta_img.exists():
            continue
        for img in sorted(pasta_img.iterdir()):
            if img.is_dir():
                continue
            txt = pasta_lbl / f"{img.stem}.txt"
            if not txt.exists():
                print(f"{AVISO}{img.name}: sem .txt correspondente, pulando")
                continue

            linhas = []
            for linha in txt.read_text(encoding="utf-8").splitlines():
                partes = linha.split()
                if len(partes) < 5:
                    continue
                antigo = int(float(partes[0]))
                novo = remap[antigo]
                contagem[export_nomes[antigo]] += 1
                linhas.append(" ".join([str(novo), *partes[1:5]]))

            if (img_dest / img.name).exists():
                sobrescritas.append(img.name)
            if not args.simular:
                shutil.copy2(img, img_dest / img.name)
                (lbl_dest / f"{img.stem}.txt").write_text(
                    "\n".join(linhas) + "\n", encoding="utf-8")
            n_img += 1
            n_lbl += 1
            n_caixas += len(linhas)
            stems.append(img.stem)

    print(f"\n== Importado ==")
    print(f"  {n_img} imagens, {n_lbl} labels, {n_caixas} caixas -> {args.raiz}")
    for nome, n in sorted(contagem.items(), key=lambda x: -x[1]):
        print(f"    {nome:<12} {n:>6} caixas")
    if sobrescritas:
        print(f"{AVISO}{len(sobrescritas)} arquivo(s) ja existiam e foram "
              f"sobrescritos: {', '.join(sobrescritas[:3])}...")

    if args.mover_originais:
        # DJI_0988_JPG.rf.XROAGx... -> DJI_0988
        origem = args.raiz / "nao_rotuladas"
        destino = args.raiz / "ja_corrigidas"
        movidas = 0
        if origem.exists():
            if not args.simular:
                destino.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                base = stem.split("_jpg.rf.")[0].split("_JPG.rf.")[0]
                for cand in origem.glob(f"{base}.*"):
                    if not args.simular:
                        shutil.move(str(cand), str(destino / cand.name))
                    movidas += 1
        print(f"\n  {movidas} original(is) -> {destino}")

    if args.simular:
        print(f"\n{AVISO}--simular: nada foi escrito.")
    else:
        print(f"\nAgora rode:  python tools/checar_dados.py")


if __name__ == "__main__":
    main()
