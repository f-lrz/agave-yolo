"""Escreve o resultado da triagem de recortes de volta nos labels.

A triagem do estagio 2 acontece movendo ARQUIVOS entre pastas. Isso e rapido
para voce, mas o resultado fica fora de data/raw/labels — ou seja, fora da
verdade de campo do projeto. Consequencias de nao fechar esse ciclo:

  * rodar make_crops.py de novo perde tudo o que voce triou;
  * o conjunto de 2 classes nunca fica completo;
  * nada downstream (medicao, retreino do detector) ve as novas doentes.

Este script le data/crops/doente/, descobre de qual caixa veio cada recorte
pelo crops_index.csv, e troca a classe daquela caixa para 'doente' no .txt.

    python tools/aplicar_triagem.py --crops data/crops --labels data/raw/labels --simular
    python tools/aplicar_triagem.py --crops data/crops --labels data/raw/labels

Faz backup antes de escrever. So mexe na PRIMEIRA coluna das linhas indicadas;
coordenada nenhuma e tocada.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

OK, AVISO, ERRO = "  ok ", "  !! ", "  XX "
EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--classe-doente", type=int, default=0)
    ap.add_argument("--classe-sadia", type=int, default=1)
    ap.add_argument("--nao-desmarcar", action="store_true",
                    help="nao reverte para sadia as caixas que voce tirou de "
                         "doente/ durante a triagem")
    ap.add_argument("--simular", action="store_true")
    args = ap.parse_args()

    idx_path = args.crops / "crops_index.csv"
    if not idx_path.exists():
        raise SystemExit(f"{ERRO}nao achei {idx_path}. Rode o make_crops.py primeiro.")

    # recorte -> (stem do label, indice da caixa)
    mapa: dict[str, tuple[str, int]] = {}
    caixas_por_stem: dict[str, int] = defaultdict(int)
    with idx_path.open(encoding="utf-8-sig") as f:
        for l in csv.DictReader(f, delimiter=";"):
            stem = Path(l["imagem_origem"]).stem
            mapa[l["recorte"]] = (stem, int(l["indice_caixa"]))
            caixas_por_stem[stem] += 1

    pasta_doente = args.crops / "doente"
    recortes = sorted(p.name for p in pasta_doente.iterdir()
                      if p.suffix.lower() in EXT) if pasta_doente.exists() else []
    if not recortes:
        raise SystemExit(f"{ERRO}nada em {pasta_doente}")

    alvos: dict[str, set[int]] = defaultdict(set)
    orfaos = []
    for nome in recortes:
        if nome not in mapa:
            orfaos.append(nome)
            continue
        stem, i = mapa[nome]
        alvos[stem].add(i)

    print(f"\n{len(recortes)} recortes em doente/ -> {sum(len(v) for v in alvos.values())} "
          f"caixas em {len(alvos)} imagens")
    if orfaos:
        print(f"{AVISO}{len(orfaos)} recorte(s) sem registro no indice (renomeado "
              f"ou de outra rodada): {', '.join(orfaos[:3])}")

    # Guarda de seguranca: o indice da caixa so vale se o .txt nao mudou desde
    # que o make_crops rodou. Conferir a contagem pega isso.
    problemas = []
    for stem, idxs in alvos.items():
        txt = args.labels / f"{stem}.txt"
        if not txt.exists():
            problemas.append(f"{stem}: .txt ausente")
            continue
        n = len([l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()])
        if n != caixas_por_stem[stem]:
            problemas.append(f"{stem}: {n} linhas agora, {caixas_por_stem[stem]} "
                             f"quando os recortes foram gerados")
        if idxs and max(idxs) >= n:
            problemas.append(f"{stem}: indice {max(idxs)} fora de {n} linhas")
    if problemas:
        print(f"\n{ERRO}os labels mudaram depois do make_crops. NAO vou escrever:")
        for p in problemas[:8]:
            print(f"       {p}")
        raise SystemExit(f"\n     Rode o make_crops.py de novo e refaca a triagem, "
                         f"ou restaure os labels daquele momento.")
    print(f"{OK}indices conferidos contra os .txt atuais")

    # A triagem e a verdade: um recorte que VOCE tirou de doente/ foi
    # reavaliado como sadio, e o label tem que acompanhar. Sem isto, uma
    # correcao sua some e a verdade de campo fica contraditoria.
    reverter: dict[str, set[int]] = defaultdict(set)
    if not args.nao_desmarcar:
        em_doente = set(recortes)
        for nome, (stem, i) in mapa.items():
            if nome in em_doente:
                continue
            txt = args.labels / f"{stem}.txt"
            if not txt.exists():
                continue
            linhas = [l for l in txt.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
            if i < len(linhas) and int(float(linhas[i].split()[0])) == args.classe_doente:
                reverter[stem].add(i)

    # Relatorio do que muda
    ja, novas = 0, 0
    for stem, idxs in alvos.items():
        linhas = [l for l in (args.labels / f"{stem}.txt").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        for i in idxs:
            if int(float(linhas[i].split()[0])) == args.classe_doente:
                ja += 1
            else:
                novas += 1
    n_rev = sum(len(v) for v in reverter.values())
    print(f"\n  ja marcadas como doente no label : {ja}")
    print(f"  passam a doente agora            : {novas}")
    if n_rev:
        print(f"  voltam a sadia (voce tirou de doente/): {n_rev}")
        print(f"       sua reavaliacao no recorte vale mais que a marcacao")
        print(f"       feita de olho na imagem inteira. Use --nao-desmarcar")
        print(f"       se preferir manter as marcacoes antigas.")

    if args.simular:
        print(f"\n{AVISO}--simular: nada escrito.")
        return

    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.labels.parent / f"{args.labels.name}_backup_{carimbo}"
    shutil.copytree(args.labels, backup)
    print(f"\n  backup -> {backup}")

    tocados = sorted(set(alvos) | set(reverter))
    for stem in tocados:
        txt = args.labels / f"{stem}.txt"
        linhas = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, classe in ([(i, args.classe_doente) for i in alvos.get(stem, ())]
                          + [(i, args.classe_sadia) for i in reverter.get(stem, ())]):
            partes = linhas[i].split()
            partes[0] = str(classe)
            linhas[i] = " ".join(partes)
        txt.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    print(f"  {novas} caixa(s) -> doente, {n_rev} -> sadia, "
          f"em {len(tocados)} arquivo(s)")
    print(f"\nConfira:  python tools/checar_dados.py")


if __name__ == "__main__":
    main()
