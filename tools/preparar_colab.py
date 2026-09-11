"""Confere o dataset de tiles e empacota para treinar no Colab.

Serve para dois casos:
  * o PC reiniciou no meio de alguma coisa e voce quer saber se sobrou dano
  * a GPU local nao e confiavel e o treino vai para a nuvem

Checa toda imagem e todo label um por um (reinicio forcado corrompe o arquivo
que estava sendo escrito naquele instante), corrige o `path:` do data.yaml — que
sai como caminho absoluto do Windows e nao existe no Colab — e gera o .zip.

    python tools/preparar_colab.py --tiles data/tiles
    python tools/preparar_colab.py --tiles data/tiles --sem-zip     # so verificar
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2

OK, AVISO, ERRO = "  ok  ", "  !!  ", "  XX  "


def checar_split(raiz: Path, split: str) -> tuple[int, int, list[str]]:
    """-> (imagens boas, caixas, problemas)"""
    d_img, d_lab = raiz / split / "images", raiz / split / "labels"
    if not d_img.exists():
        return 0, 0, [f"pasta {split}/images nao existe"]

    problemas, boas, caixas = [], 0, 0
    imgs = sorted(d_img.glob("*.jpg"))

    # label sem imagem: nao apareceria iterando so pelas imagens
    nomes = {p.stem for p in imgs}
    for lab in sorted(d_lab.glob("*.txt")):
        if lab.stem not in nomes:
            problemas.append(f"{split}/images/{lab.stem}.jpg: sumiu (label existe)")

    for p in imgs:
        # JPG truncado NAO faz o imread devolver None — ele devolve a parte que
        # conseguiu decodificar. O teste confiavel e o marcador de fim de arquivo
        # (FFD9), que so existe se o JPEG foi gravado por inteiro.
        dados = p.read_bytes()
        if len(dados) < 4 or dados[-2:] != b"\xff\xd9":
            problemas.append(f"{split}/images/{p.name}: truncado (sem fim de JPEG)")
            continue
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            problemas.append(f"{split}/images/{p.name}: ilegivel")
            continue
        lab = d_lab / f"{p.stem}.txt"
        if not lab.exists():
            problemas.append(f"{split}/labels/{p.stem}.txt: faltando")
            continue
        try:
            linhas = lab.read_text().splitlines()
        except Exception as e:
            problemas.append(f"{split}/labels/{p.stem}.txt: {e}")
            continue
        for i, linha in enumerate(linhas, 1):
            partes = linha.split()
            if not partes:
                continue
            if len(partes) != 5:
                problemas.append(f"{split}/labels/{p.stem}.txt linha {i}: "
                                 f"{len(partes)} colunas")
                break
            try:
                v = [float(x) for x in partes]
            except ValueError:
                problemas.append(f"{split}/labels/{p.stem}.txt linha {i}: "
                                 f"valor invalido")
                break
            if any(x < 0 or x > 1 for x in v[1:]):
                problemas.append(f"{split}/labels/{p.stem}.txt linha {i}: "
                                 f"fora de 0-1")
                break
            caixas += 1
        else:
            boas += 1
    return boas, caixas, problemas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=Path, default=Path("data/tiles"))
    ap.add_argument("--zip", dest="destino", type=Path, default=None,
                    help="caminho do .zip (padrao: <tiles>.zip)")
    ap.add_argument("--sem-zip", action="store_true", help="so verificar")
    args = ap.parse_args()

    if not args.tiles.exists():
        raise SystemExit(f"{args.tiles} nao existe — rode o make_tiles.py antes")

    print(f"\n== Verificando {args.tiles} ==")
    total_img = total_cx = 0
    falhou = False
    for split in ("train", "val", "test"):
        boas, caixas, problemas = checar_split(args.tiles, split)
        total_img += boas
        total_cx += caixas
        if problemas:
            falhou = True
            print(ERRO + f"{split}: {len(problemas)} problema(s)")
            for pr in problemas[:5]:
                print(f"        {pr}")
            if len(problemas) > 5:
                print(f"        ... e mais {len(problemas) - 5}")
        else:
            print(OK + f"{split}: {boas} tiles, {caixas} caixas")

    if falhou:
        print("\n" + ERRO + "Ha arquivos corrompidos. Nao tente consertar um por um —")
        print("       apague data/tiles e rode o make_tiles.py de novo (os dados")
        print("       originais em data/raw estao intactos, o tiles e derivado).")
        return

    classes = set()
    for lab in args.tiles.rglob("labels/*.txt"):
        for linha in lab.read_text().splitlines():
            if linha.split():
                classes.add(linha.split()[0])
    print(OK + f"classes presentes: {sorted(classes)}")
    if len(classes) > 1:
        print(AVISO + "mais de uma classe nos tiles. Se o data.yaml declara so uma,")
        print("       o Ultralytics DESCARTA os tiles da classe extra em silencio.")
        print("       Rode o juntar_classes.py e refaca os tiles.")

    # O path: sai como caminho absoluto do Windows e nao existe no Colab.
    # Relativo funciona nos dois: o Ultralytics resolve a partir do data.yaml.
    yml = args.tiles / "data.yaml"
    if yml.exists():
        linhas = yml.read_text(encoding="utf-8").splitlines()
        novas = ["path: ." if l.startswith("path:") else l for l in linhas]
        if novas != linhas:
            yml.write_text("\n".join(novas) + "\n", encoding="utf-8")
            print(OK + "data.yaml: 'path:' trocado para relativo (funciona local e no Colab)")

    print(f"\n{total_img} tiles, {total_cx} caixas — dataset integro.")

    if args.sem_zip:
        return
    destino = args.destino or args.tiles.with_suffix(".zip")
    print(f"\nCompactando para {destino}...")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(args.tiles.rglob("*")):
            if f.is_file() and f.suffix != ".zip":
                z.write(f, f.relative_to(args.tiles.parent))
    mb = destino.stat().st_size / 1024 ** 2
    print(OK + f"{destino}  ({mb:.0f} MB)")
    print("\nSuba esse .zip no Colab (ou no Google Drive) e rode o colab_treino.ipynb.")


if __name__ == "__main__":
    main()