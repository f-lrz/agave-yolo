"""Detector de agaves por visao computacional classica — SEM treinar nada.

Explora tres propriedades do seu problema:
  1. agave e verde, solo e marrom/cinza  -> indice de vegetacao separa os dois
  2. a roseta e um blob aproximadamente circular e solido -> filtro de forma
  3. plantas encostadas viram um blob so -> watershed sobre a transformada de
     distancia separa (o centro da roseta e o ponto mais distante da borda)

O resultado sao PROPOSTAS, nao labels finais. O objetivo e voce corrigir em vez
de desenhar do zero. Espere algo entre 70% e 90% de recall numa imagem limpa;
mato, sombra e plantas sobrepostas derrubam isso.

    # 1. ache os parametros olhando o diagnostico
    python src/cv_proposals.py --images data/raw/amostra --out out/cv --debug

    # 2. rode em tudo, gerando labels YOLO
    python src/cv_proposals.py --images data/raw/nao_rotuladas --out data/preanot \\
        --plant-px 130 --save-labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tiling import list_images, to_yolo_lines

# --------------------------------------------------------------------------
# indices de vegetacao
# --------------------------------------------------------------------------

def _norm_rgb(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB normalizado pelo brilho. Isso e o que da robustez a sombra: uma folha
    na sombra tem menos luz, mas a PROPORCAO de verde continua alta."""
    b, g, r = cv2.split(img.astype(np.float32))
    s = b + g + r + 1e-6
    return b / s, g / s, r / s


def vegetation_index(img: np.ndarray, kind: str) -> np.ndarray:
    """Retorna um mapa uint8 onde vegetacao e clara e solo e escuro."""
    B, G, R = _norm_rgb(img)
    if kind == "exg":            # Excess Green — o mais usado em agricultura
        v = 2 * G - R - B
    elif kind == "exgr":         # ExG menos Excess Red; melhor com solo avermelhado
        v = (2 * G - R - B) - (1.4 * R - G)
    elif kind == "vari":         # tolera variacao de iluminacao
        v = (G - R) / (G + R - B + 1e-6)
    elif kind == "hsv":          # matiz puro; util quando o solo e esverdeado
        h, s, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32))
        v = (np.abs(((h - 60) + 90) % 180 - 90) < 30).astype(np.float32) * (s / 255.0)
    elif kind == "lab_a":
        # Eixo verde-vermelho do CIELAB. Separa VERDE de AMARELO, coisa que o
        # ExG nao faz: capim seco tem G > R e passa como vegetacao no ExG, mas
        # fica do lado amarelo no a*. E o indice para fundo com mato seco.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        v = 128.0 - lab[:, :, 1]
    elif kind == "verde_escuro":
        # Agave: verde saturado E escuro. Capim seco: pouco saturado e claro.
        # Multiplicar as tres condicoes derruba o capim mesmo quando ele e verde.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        verde = np.clip(128.0 - lab[:, :, 1], 0, None) / 40.0
        sat = hsv[:, :, 1] / 255.0
        escuro = 1.0 - hsv[:, :, 2] / 255.0
        v = verde * sat * (0.4 + 0.6 * escuro)
    elif kind == "gli":          # Green Leaf Index, normalizado
        v = (2 * G - R - B) / (2 * G + R + B + 1e-6)
    else:
        raise ValueError(f"indice desconhecido: {kind}")
    v = np.nan_to_num(v)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    return np.clip((v - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# estimativa automatica do tamanho da planta
# --------------------------------------------------------------------------

def estimate_plant_px(mask: np.ndarray) -> float:
    """Diametro tipico dos blobs de vegetacao, em pixels.

    Mediana PONDERADA POR AREA, nao mediana simples. A diferenca importa muito:
    um talhao com 50 agaves costuma ter centenas de fragmentos de mato, que
    dominam a contagem e puxam a mediana simples para o tamanho do mato. Como
    os agaves concentram quase toda a AREA de vegetacao, ponderar por area cai
    na populacao certa.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])
    areas = areas[areas > 60]
    if len(areas) < 5:
        return 0.0
    cum = np.cumsum(areas) / areas.sum()
    return float(np.sqrt(4 * areas[int(np.searchsorted(cum, 0.5))] / np.pi))


def calibrate_plant_px(mask: np.ndarray, n_iter: int = 3) -> float:
    """Refina a estimativa iterando estimar -> fechar -> reestimar.

    A primeira medida sai baixa porque a roseta ainda esta em pedacos (as folhas
    tem vao entre elas). Fechando com base nela, as folhas se unem e a proxima
    medida sobe. Converge em 2-3 voltas.
    """
    d = estimate_plant_px(mask)
    if d <= 0:
        return 0.0
    # Teto de sanidade. Num plantio denso a mascara vira um lencol continuo, o
    # fechamento funde tudo, e a estimativa dispara — ja vi ela chegar a 6x o
    # tamanho real e produzir 13 caixas na imagem inteira. Nenhuma planta ocupa
    # 1/6 do lado da foto.
    teto = min(mask.shape[:2]) / 6.0
    for _ in range(n_iter):
        fechado = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(d * 0.22))
        novo = estimate_plant_px(fechado)
        if novo <= d * 1.03:          # estabilizou
            break
        d = min(novo, d * 2.0, teto)  # trava crescimento explosivo por fusao
    return min(d, teto)


# --------------------------------------------------------------------------
# nucleo
# --------------------------------------------------------------------------

def _ellipse(k: int) -> np.ndarray:
    k = max(3, int(k) | 1)                      # sempre impar e >= 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def segment(img: np.ndarray, index: str, thresh: int | None, plant_px: float | None,
            thresh_scale: float = 0.40, abertura: float = 0.04,
            fechamento: float = 0.22) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (mapa do indice, mascara binaria limpa, plant_px usado)

    Sobre o limiar: o Otsu puro e agressivo DEMAIS para este problema. Ele
    assume duas populacoes (solo e planta) e corta no vale entre elas — mas a
    planta amarelada/doente cai justamente NO vale, e some. Como o objetivo
    aqui e pre-anotar (recall vale mais que precisao: apagar caixa e rapido,
    desenhar a que faltou e lento), o padrao e cortar bem abaixo do Otsu.
    """
    veg = vegetation_index(img, index)

    if thresh is None:
        otsu, _ = cv2.threshold(veg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh_scale > 1 e legitimo: em plantio denso o Otsu ainda deixa solo
        # entrar. O clamp evita que um valor alto zere a mascara inteira.
        thresh = min(otsu * thresh_scale, 254.0)
    _, mask = cv2.threshold(veg, float(thresh), 255, cv2.THRESH_BINARY)

    if not plant_px:
        rough = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(3))
        plant_px = calibrate_plant_px(rough)
        if plant_px <= 0:
            plant_px = min(img.shape[:2]) / 30.0    # chute de ultimo caso

    # Os dois parametros que decidem o sucesso quando o fundo NAO e solo limpo:
    #
    #   abertura   mata o ruido de fundo (capim, mato) antes de qualquer colagem.
    #              Precisa ser maior que a espessura do ruido e menor que a
    #              espessura da folha do agave.
    #   fechamento une as folhas da roseta, que tem vao entre elas. Mas cola
    #              TUDO que estiver a essa distancia: se sobrou ruido de fundo,
    #              um fechamento grande transforma a imagem inteira num lencol
    #              continuo e a fronteira das plantas desaparece.
    #
    # Com fundo de terra os padroes (0.04 / 0.22) funcionam. Com capim no chao,
    # quase sempre e preciso abrir mais e fechar menos.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(plant_px * abertura))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(plant_px * fechamento))
    return veg, mask, plant_px


def find_peaks(dist: np.ndarray, plant_px: float, peak_sep: float = 0.45) -> np.ndarray:
    """Centros de planta: maximos locais da transformada de distancia.

    O centro da roseta e o ponto mais distante da borda, entao vira um pico.
    Duas plantas encostadas formam um blob unico com DOIS picos.

    `peak_sep` e a distancia minima entre dois centros, em fracao de plant_px, e
    e o parametro que mais mexe no recall: alto demais funde duas plantas
    vizinhas num pico so (voce perde uma); baixo demais parte uma roseta em
    duas (voce ganha uma caixa falsa).
    """
    smooth = cv2.GaussianBlur(dist, (0, 0), max(plant_px * 0.08, 1.0))
    k = _ellipse(plant_px * peak_sep)
    peaks = (smooth >= cv2.dilate(smooth, k) - 1e-4) & (smooth > plant_px * 0.12)
    return peaks.astype(np.uint8)


def split_touching(mask: np.ndarray, plant_px: float, max_radius_ratio: float = 0.7,
                   peak_sep: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Reparte a mascara entre os picos, por proximidade -> (rotulos, distancia).

    Cada pixel de vegetacao vai para o pico mais proximo (Voronoi). Para duas
    rosetas encostadas, o corte cai exatamente no meio — que e o certo.

    Preferido ao cv2.watershed aqui por dois motivos praticos: cobre a mascara
    inteira (o watershed deixava o fundo invadir a planta, porque a borda
    planta/solo nao forma cume na superficie de distancia, e as caixas saiam so
    com o miolo da roseta), e nao depende da convencao de marcadores.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = find_peaks(dist, plant_px, peak_sep)
    if peaks.max() == 0:
        return cv2.connectedComponents(mask, 8)[1], dist

    # distancia ate o pico mais proximo + rotulo desse pico, numa chamada so
    dist_ao_pico, labels = cv2.distanceTransformWithLabels(
        (1 - peaks) * 255, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)

    # O limite de raio faz dois trabalhos: impede que uma planta absorva o mato
    # vizinho, e descarta blobs de mato que nao tem pico nenhum dentro (senao
    # seriam adotados por uma planta distante, inflando a caixa dela).
    valido = (mask > 0) & (dist_ao_pico <= max_radius_ratio * plant_px)
    return (labels * valido).astype(np.int32), dist


def boxes_from_labels(markers: np.ndarray, plant_px: float, min_area_ratio: float,
                      max_area_ratio: float, min_solidity: float, max_aspect: float,
                      classe: int = 0) -> tuple[np.ndarray, dict[str, int]]:
    """Cada regiao vira uma caixa, se passar nos filtros de forma.

    Area e caixa saem de UMA varredura da imagem, e nao de um `markers == lab`
    por regiao. A versao ingenua era O(regioes x pixels): com 550 plantas numa
    imagem de 16 MP dava 9 bilhoes de comparacoes, ~100 s por imagem. O contorno
    (para a solidez) roda so dentro da caixa da regiao, que e minuscula.
    """
    nominal = np.pi / 4 * plant_px ** 2
    lo, hi = min_area_ratio * nominal, max_area_ratio * nominal
    boxes: list[list[float]] = []
    rej = {"area": 0, "solidez": 0, "alongado": 0}

    ys, xs = np.nonzero(markers)
    if len(ys) == 0:
        return np.zeros((0, 5)), rej

    labs = markers[ys, xs]
    ordem = np.argsort(labs, kind="stable")
    labs, xs, ys = labs[ordem], xs[ordem], ys[ordem]
    n = int(labs[-1]) + 1
    inicio = np.searchsorted(labs, np.arange(n + 1))

    for lab in range(1, n):
        a, b = inicio[lab], inicio[lab + 1]
        area = b - a
        if area == 0:
            continue
        if area < lo or area > hi:
            rej["area"] += 1
            continue
        bx1, bx2 = int(xs[a:b].min()), int(xs[a:b].max())
        by1, by2 = int(ys[a:b].min()), int(ys[a:b].max())
        w, h = bx2 - bx1 + 1, by2 - by1 + 1
        if max(w, h) / max(min(w, h), 1) > max_aspect:
            rej["alongado"] += 1
            continue
        sub = (markers[by1:by2 + 1, bx1:bx2 + 1] == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        # solidez separa roseta (compacta) de mato/capim (esparramado)
        if hull > 0 and cv2.contourArea(c) / hull < min_solidity:
            rej["solidez"] += 1
            continue
        boxes.append([classe, bx1, by1, bx2 + 1, by2 + 1])

    return np.asarray(boxes, dtype=np.float64).reshape(-1, 5), rej


def detect(img: np.ndarray, args) -> tuple[np.ndarray, dict]:
    """Detecta na imagem reduzida e devolve as caixas na escala ORIGINAL.

    Segmentar em 5472 px uma planta de 300 px e desperdicio: nada aqui depende
    de detalhe fino, e o custo da morfologia cresce com o tamanho do kernel, que
    e proporcional ao tamanho da planta. Reduzindo para ~2000 px de lado maior,
    o fechamento cai de 0.42 s para 0.01 s sem perder nenhuma planta.
    """
    h0, w0 = img.shape[:2]
    max_lado = getattr(args, "max_lado", 2000) or 0
    esc = 1.0
    trab = img
    if max_lado and max(h0, w0) > max_lado:
        esc = max_lado / max(h0, w0)
        trab = cv2.resize(img, (max(int(w0 * esc), 1), max(int(h0 * esc), 1)),
                          interpolation=cv2.INTER_AREA)

    ppx = args.plant_px * esc if args.plant_px else None
    veg, mask, plant_px = segment(trab, args.index, args.thresh, ppx, args.thresh_scale,
                                  getattr(args, "abertura", 0.04),
                                  getattr(args, "fechamento", 0.22))
    markers, dist = split_touching(mask, plant_px, args.max_radius, args.peak_sep)
    boxes, rej = boxes_from_labels(markers, plant_px, args.min_area, args.max_area,
                                   args.min_solidity, args.max_aspect, args.classe)
    if esc != 1.0 and len(boxes):
        boxes[:, 1:5] /= esc                  # de volta para a imagem original

    return boxes, {"plant_px": plant_px / esc, "veg": veg, "mask": mask, "dist": dist,
                   "markers": markers, "rejeitados": rej, "escala": esc,
                   "trabalho": trab, "cobertura": float((mask > 0).mean())}


# --------------------------------------------------------------------------
# saidas
# --------------------------------------------------------------------------

def draw(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(img.shape[:2]) / 900))
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), t)
    return out


def debug_panel(img: np.ndarray, info: dict, boxes: np.ndarray, dest: Path) -> None:
    """Painel 2x3 com cada etapa. E aqui que voce descobre ONDE esta errando:
    se a mascara ja perdeu a planta, mexer nos filtros de forma nao adianta."""
    h = 520
    def prep(x, titulo, cor=False):
        if x.ndim == 2:
            x = cv2.applyColorMap(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLORMAP_VIRIDIS) if cor else \
                cv2.cvtColor(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
                             .astype(np.uint8), cv2.COLOR_GRAY2BGR)
        s = h / x.shape[0]
        x = cv2.resize(x, (int(x.shape[1] * s), h))
        cv2.rectangle(x, (0, 0), (x.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(x, titulo, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return x

    mk = (info["markers"] > 0).astype(np.uint8) * 255
    linha1 = np.hstack([prep(img, "1. original"),
                        prep(info["veg"], "2. indice de vegetacao"),
                        prep(info["mask"], "3. mascara pos-morfologia")])
    linha2 = np.hstack([prep(info["dist"], "4. transformada de distancia", cor=True),
                        prep(mk, "5. regioes apos watershed"),
                        prep(draw(img, boxes), f"6. {len(boxes)} propostas")])
    cv2.imwrite(str(dest), np.vstack([linha1, linha2]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser(description="propostas de agave sem treinar modelo")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", default="exg", choices=["exg", "exgr", "vari", "hsv", "lab_a", "verde_escuro", "gli"],
                    help="exg para solo nu; lab_a ou verde_escuro quando o "
                         "fundo tem capim/mato (rode tools/escolher_indice.py)")
    ap.add_argument("--thresh", type=int, default=None,
                    help="limiar fixo 0-255; o padrao e Otsu automatico")
    ap.add_argument("--thresh-scale", type=float, default=0.40,
                    help="fracao do limiar de Otsu a usar (padrao 0.45). Menor = "
                         "pega mais planta amarelada/doente, ao custo de mais mato")
    ap.add_argument("--plant-px", type=float, default=None,
                    help="diametro tipico do agave em px; vazio = estimar sozinho")
    ap.add_argument("--min-area", type=float, default=0.20,
                    help="area minima, como fracao da area nominal da planta")
    ap.add_argument("--max-area", type=float, default=3.0)
    ap.add_argument("--min-solidity", type=float, default=0.55,
                    help="area/area do fecho convexo; baixo demais aceita mato")
    ap.add_argument("--max-aspect", type=float, default=2.5)
    ap.add_argument("--peak-sep", type=float, default=0.45,
                    help="distancia minima entre centros, em fracao de plant_px; "
                         "o parametro que mais mexe no recall")
    ap.add_argument("--max-radius", type=float, default=0.7,
                    help="raio maximo de uma planta, como fracao de plant_px; "
                         "limita ate onde uma regiao pode crescer a partir do centro")
    ap.add_argument("--abertura", type=float, default=0.04,
                    help="raio da abertura, em fracao de plant_px. SUBA quando o "
                         "fundo tem capim/mato: mata o ruido antes que ele grude")
    ap.add_argument("--fechamento", type=float, default=0.22,
                    help="raio do fechamento, em fracao de plant_px. BAIXE quando a "
                         "mascara vira um lencol continuo: e ele que cola tudo")
    ap.add_argument("--max-lado", type=int, default=2000,
                    help="reduz a imagem para no maximo este lado antes de segmentar "
                         "(as caixas voltam para a escala original). 0 = nao reduzir. "
                         "Deixe a planta com ~80-150px na imagem reduzida")
    ap.add_argument("--classe", type=int, default=0,
                    help="id de classe escrito nos .txt. Se o seu projeto de "
                         "anotacao usa 0=doente e 1=sadia, use --classe 1: quase "
                         "toda proposta esta certa assim, e voce so vira as poucas "
                         "doentes na correcao")
    ap.add_argument("--save-labels", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="salva o painel com todas as etapas intermediarias")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    imgs = list_images(args.images)
    if not imgs:
        raise SystemExit(f"nenhuma imagem em {args.images}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    if args.save_labels:
        (args.out / "labels").mkdir(parents=True, exist_ok=True)
    if args.debug:
        (args.out / "debug").mkdir(parents=True, exist_ok=True)

    total, rej_tot, plant_pxs = 0, {"area": 0, "solidez": 0, "alongado": 0}, []
    coberturas: list[float] = []

    for p in tqdm(imgs, desc="propostas"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes, info = detect(img, args)
        total += len(boxes)
        plant_pxs.append(info["plant_px"])
        coberturas.append(info["cobertura"])
        for k, v in info["rejeitados"].items():
            rej_tot[k] += v

        if not args.no_preview:
            cv2.imwrite(str(args.out / "images" / f"{p.stem}.jpg"), draw(img, boxes),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        if args.save_labels:
            (args.out / "labels" / f"{p.stem}.txt").write_text(
                "\n".join(to_yolo_lines(boxes, w, h)) + "\n")
        if args.debug:
            debug_panel(img, info, boxes, args.out / "debug" / f"{p.stem}.jpg")

    print(f"\n{total} propostas em {len(imgs)} imagens "
          f"({total / max(len(imgs), 1):.1f} por imagem)")
    if plant_pxs:
        print(f"plant_px usado: {np.mean(plant_pxs):.0f} "
              f"(min {min(plant_pxs):.0f}, max {max(plant_pxs):.0f})")
        if args.plant_px is None:
            print("  -> se esse valor nao bate com o tamanho real do agave, "
                  "passe --plant-px voce mesmo: ele calibra todos os outros filtros")
    print(f"regioes descartadas: {rej_tot['area']} por area, "
          f"{rej_tot['solidez']} por solidez, {rej_tot['alongado']} por serem alongadas")

    cob = float(np.mean(coberturas)) if coberturas else 0.0
    print(f"vegetacao ocupa {cob * 100:.1f}% da imagem")
    if cob > 0.70:
        print("  !! ALTO DEMAIS. O limiar inundou a mascara: o solo entrou como")
        print("     vegetacao e as plantas viraram um borrao so. Suba --thresh-scale")
        print("     (tente 0.5, 0.6).")
    elif cob > 0.35:
        print("  ?  Cobertura alta. Pode ser mascara inundada, ou pode ser um")
        print("     plantio denso de verdade — este numero sozinho nao distingue.")
        print("     Se voce tem imagens rotuladas, meca em vez de adivinhar:")
        print("       python tools/varrer_parametros.py --param thresh_scale "
              "--valores 0.3 0.4 0.55 0.7")
    elif cob < 0.005:
        print("  !! BAIXO DEMAIS. Quase nada passou no limiar. Baixe --thresh-scale,")
        print("     ou troque --index (exgr se o solo for avermelhado).")
    if args.debug:
        print(f"\nOlhe {args.out / 'debug'} — o painel mostra em qual etapa voce "
              f"esta perdendo planta.")


if __name__ == "__main__":
    main()
