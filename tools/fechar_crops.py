"""ETAPA 2 — fecha a triagem e monta o dataset de treino do classificador.

Roda depois que voce arrastou as doentes para data/crops/doente/.
O que sobrou em _classificar/ e tratado como sadia.

    python tools/fechar_crops.py --crops data/crops
    python tools/fechar_crops.py --crops data/crops --simular   # so relatorio

O split treino/validacao e feito POR IMAGEM DE ORIGEM, nunca por recorte.
Recortes da mesma foto compartilham iluminacao, altura de voo e terreno — se
voce embaralhar recortes, a acuracia fica linda e o modelo nao generaliza. E a
mesma licao do make_tiles.py na etapa 1.

Alem de montar as pastas, ele diz se ha exemplos suficientes para treinar. E a
pergunta que importa nesta etapa: com desbalanceamento extremo, um modelo que
responde 'sadia' sempre acerta 99,5% e nao serve para nada.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

OK, AVISO, ERRO = "  ok ", "  !! ", "  XX "
EXT = {".jpg", ".jpeg", ".png"}

# Minimos para o treino fazer sentido. Nao sao regras universais — sao o ponto
# a partir do qual um classificador binario com transfer learning costuma sair
# do chute em dataset pequeno e desbalanceado.
MIN_DOENTE_TREINO = 60
MIN_DOENTE_VAL = 15


def stem_origem(nome: str) -> str:
    """'0042_DJI_0991_jpg.rf.abc__0137.jpg' -> 'DJI_0991_jpg.rf.abc'.

    Remove o prefixo do ranking (digitos + '_') e o sufixo '__<indice>'.
    """
    base = Path(nome).stem
    if "__" in base:
        base = base.rsplit("__", 1)[0]
    partes = base.split("_", 1)
    if len(partes) == 2 and partes[0].isdigit():
        base = partes[1]
    return base


def listar(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXT)


def rank_do_nome(nome: str) -> int:
    """'00263_DJI_0991__0137.jpg' -> 263. Sem prefixo, devolve um numero alto."""
    primeiro = Path(nome).stem.split("_", 1)[0]
    return int(primeiro) if primeiro.isdigit() else 10**9


def subamostra_sadias(sadias: list[Path], n_doentes: int, args) -> list[Path]:
    """Reduz a classe sadia, metade por sorteio e metade do topo do ranking.

    Por que subamostrar: com 259 sadias por doente, o modelo minimiza a perda
    aprendendo a responder 'sadia' sempre. Peso de classe ajuda, mas subamostrar
    tambem encurta muito o treino — e aqui a GPU nao e confiavel.

    Por que METADE do topo do ranking: aquelas sao as sadias que a cor confundiu
    com doente, ou seja, os negativos DIFICEIS — plantas amareladas por sol,
    poeira ou sombra que nao estao doentes. Sao elas que ensinam a fronteira. Se
    todas as sadias vierem de sorteio, a maioria sera verde obvia e o modelo
    aprende uma tarefa mais facil do que a real.
    """
    if args.max_sadias_por_doente <= 0:
        return sadias
    alvo = n_doentes * args.max_sadias_por_doente
    if alvo >= len(sadias):
        return sadias

    n_dif = int(alvo * args.frac_dificeis)
    por_rank = sorted(sadias, key=lambda p: rank_do_nome(p.name))
    dificeis = por_rank[:n_dif]

    restantes = por_rank[n_dif:]
    rng = random.Random(args.seed)
    faceis = rng.sample(restantes, min(alvo - n_dif, len(restantes)))

    print(f"\n== Subamostragem da classe sadia ==")
    print(f"  {len(sadias)} sadias -> {len(dificeis) + len(faceis)} "
          f"({args.max_sadias_por_doente} por doente)")
    print(f"    {len(dificeis)} do topo do ranking (negativos dificeis)")
    print(f"    {len(faceis)} sorteadas do resto")
    print(f"  (use --max-sadias-por-doente 0 para usar todas as {len(sadias)})")
    return sorted(dificeis + faceis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, required=True,
                    help="pasta gerada pelo make_crops.py")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sadias-por-doente", type=int, default=10,
                    help="subamostra a classe sadia nesta proporcao. 0 = usa "
                         "todas (desbalanceamento extremo, e treino lento)")
    ap.add_argument("--frac-dificeis", type=float, default=0.5,
                    help="fracao das sadias que vem do TOPO do ranking de "
                         "suspeita (negativos dificeis) em vez de sorteio")
    ap.add_argument("--simular", action="store_true",
                    help="mostra o relatorio sem criar as pastas de treino")
    ap.add_argument("--out", type=Path, default=None,
                    help="onde gravar train/ e val/ (padrao: a propria --crops). "
                         "Use para nao sobrescrever um split existente.")
    ap.add_argument("--sem-val", action="store_true",
                    help="modo PRODUCAO: poe TODOS os dados no treino (sem holdout "
                         "limpo). Cria um val pequeno, duplicado do treino, so para "
                         "o YOLO validar e salvar best.pt — ele NAO mede "
                         "generalizacao. As metricas reais vem da avaliacao feita "
                         "antes, num holdout limpo.")
    args = ap.parse_args()
    out = args.out or args.crops

    doentes = listar(args.crops / "doente")
    sadias = listar(args.crops / "_classificar")
    if not doentes and not sadias:
        raise SystemExit(f"nada em {args.crops}. Rode o make_crops.py primeiro.")

    print(f"\n== Triagem ==")
    print(f"  doente        : {len(doentes)}")
    print(f"  sadia (resto) : {len(sadias)}")
    total = len(doentes) + len(sadias)
    print(f"  prevalencia   : {100 * len(doentes) / total:.2f}%  "
          f"(1 doente a cada {total // max(len(doentes), 1)} plantas)")

    sadias = subamostra_sadias(sadias, len(doentes), args)

    # Split por imagem de origem. As doentes mandam no split: sao a classe
    # rara, e e nelas que a validacao precisa ter exemplo suficiente.
    por_img_d: dict[str, list[Path]] = {}
    for p in doentes:
        por_img_d.setdefault(stem_origem(p.name), []).append(p)

    imgs_com_doente = sorted(por_img_d)
    rng = random.Random(args.seed)
    rng.shuffle(imgs_com_doente)

    val_imgs: set[str] = set()
    alvo = max(1, round(len(doentes) * args.val_frac))
    acumulado = 0
    for nome_img in imgs_com_doente:
        if acumulado >= alvo:
            break
        val_imgs.add(nome_img)
        acumulado += len(por_img_d[nome_img])

    if args.sem_val:
        val_imgs = set()   # modo producao: nada segurado para validacao

    def destino(p: Path, classe: str) -> tuple[str, str]:
        split = "val" if stem_origem(p.name) in val_imgs else "train"
        return split, classe

    plano = [(p, *destino(p, "doente")) for p in doentes]
    plano += [(p, *destino(p, "sadia")) for p in sadias]
    contagem = Counter((s, c) for _, s, c in plano)

    print(f"\n== Split por imagem de origem ==")
    print(f"  imagens com doente : {len(imgs_com_doente)}")
    print(f"  imagens na validacao: {len(val_imgs)}")
    for split in ("train", "val"):
        d, s = contagem[(split, "doente")], contagem[(split, "sadia")]
        print(f"  {split:<5} doente {d:>5}   sadia {s:>6}")

    # Veredito
    d_tr, d_va = contagem[("train", "doente")], contagem[("val", "doente")]
    print(f"\n== Da para treinar? ==")
    if args.sem_val:
        print(f"{OK}modo producao: {len(doentes)} doentes e {len(sadias)} sadias, "
              f"tudo no treino.")
        print(f"     O val e duplicado do treino, so para o YOLO rodar — nao mede")
        print(f"     generalizacao. As metricas reais vem do holdout avaliado antes.")
    elif d_tr < MIN_DOENTE_TREINO:
        print(f"{ERRO}so {d_tr} doentes no treino (minimo pratico ~{MIN_DOENTE_TREINO}).")
        print(f"     Treinar agora produz um modelo que responde 'sadia' sempre")
        print(f"     e acerta {100 * (1 - len(doentes) / total):.1f}% sem aprender nada.")
        print(f"     Continue a triagem, ou converse com o cliente sobre um voo")
        print(f"     dirigido as areas com planta doente.")
    elif d_va < MIN_DOENTE_VAL:
        print(f"{AVISO}{d_va} doentes na validacao e pouco para medir (minimo ~{MIN_DOENTE_VAL}).")
        print(f"     Da para treinar, mas a metrica vai oscilar muito.")
    else:
        print(f"{OK}{d_tr} no treino e {d_va} na validacao. Da para tentar.")

    razao = contagem[("train", "sadia")] / max(d_tr, 1)
    if razao > 20:
        print(f"\n{AVISO}desbalanceamento no treino: {razao:.0f} sadias por doente.")
        print(f"     Reduza com --max-sadias-por-doente, senao o modelo aprende")
        print(f"     a ignorar a classe doente.")
    else:
        print(f"\n{OK}balanceamento: {razao:.0f} sadias por doente no treino.")

    if args.simular:
        print(f"\n{AVISO}--simular: nada foi criado.")
        return

    for split in ("train", "val"):
        for classe in ("sadia", "doente"):
            (out / split / classe).mkdir(parents=True, exist_ok=True)
    for p, split, classe in plano:
        shutil.copy2(p, out / split / classe / p.name)

    if args.sem_val:
        # val descartavel: amostra duplicada do treino, so para o YOLO validar e
        # salvar best.pt. Nao e holdout — mede o proprio treino.
        rng_v = random.Random(args.seed + 1)
        for classe, itens in (("doente", doentes), ("sadia", sadias)):
            k = max(1, round(len(itens) * args.val_frac))
            for p in rng_v.sample(itens, min(k, len(itens))):
                shutil.copy2(p, out / "val" / classe / p.name)

    print(f"\n  dataset -> {out / 'train'} e {out / 'val'}")
    print(f"  (copiado, nao movido: _classificar/ e doente/ continuam intactos)")
    print(f"\nTreinar:  python src/train_cls.py --data {out} --epochs 60")


if __name__ == "__main__":
    main()
