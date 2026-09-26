"""Detector de agave em imagem de drone.

O QUE ELE FAZ: recebe uma foto aerea e devolve uma caixa por planta de agave.
O QUE ELE NAO FAZ: nao diz se a planta esta doente ou sadia. O modelo tem UMA
classe ('agave'). Qualquer tela que rotule a saida como "doente" esta errada.

Por que existe esta classe em vez de chamar o YOLO direto:

    # ERRADO — encontra quase nada
    YOLO('best.pt')('foto.jpg')

Uma foto de drone tem 5472x3078. O YOLO redimensiona a entrada para 1024, o
que faz um agave de ~130px virar ~24px e desaparecer. A imagem TEM que ser
recortada em tiles, cada tile inferido separadamente, e os resultados
remontados com NMS global. E isso que esta classe faz.

    from agave_detector import DetectorAgave

    det = DetectorAgave('modelo/best.pt')   # carregue UMA vez, no startup
    r = det.detectar('foto.jpg')
    print(r['total'], r['caixas'][0])
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from .tiling import tile_grid

# Contrato do modelo. Estes valores foram medidos, nao escolhidos por gosto —
# ver README.md, secao "Parametros". Mudar qualquer um degrada o resultado.
TILE = 1024
OVERLAP = 0.4
IMGSZ = 1024
IOU_NMS = 0.5
CONF_PADRAO = 0.30      # calibrado para CONTAGEM (erro de +4,9%)
CONF_PRE_ANOTACAO = 0.20  # so para gerar rascunho de rotulo, nao para exibir

# Abaixo disto a foto provavelmente foi redimensionada no caminho, e o modelo
# vai falhar em silencio: as plantas ficam pequenas demais para ele.
LARGURA_MINIMA_ESPERADA = 3000

Imagem = Union[str, Path, bytes, bytearray, np.ndarray]

# BGR (e OpenCV, nao RGB). Preto por padrao: verde se confunde com a propria
# planta na foto.
PRETO = (0, 0, 0)
MAGENTA = (255, 0, 255)
CIANO = (255, 255, 0)


def ler_imagem(imagem: Imagem) -> np.ndarray:
    """Aceita caminho, bytes (como vem de um upload) ou array numpy -> BGR."""
    if isinstance(imagem, np.ndarray):
        return imagem
    if isinstance(imagem, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(imagem, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bytes recebidos nao sao uma imagem valida")
        return img
    img = cv2.imread(str(imagem))
    if img is None:
        raise ValueError(f"nao consegui ler a imagem: {imagem}")
    return img


def desenhar_caixas(imagem: Imagem, caixas: list[dict[str, Any]],
                    cor: tuple[int, int, int] = PRETO) -> np.ndarray:
    """Desenha as caixas sobre a imagem. Devolve BGR, pronto para cv2.imencode.

    **Nao carrega o modelo.** E esta a funcao da rota de download: guarde as
    caixas que o /detectar devolveu e redesenhe a partir delas, em vez de rodar
    a deteccao outra vez (que custaria os ~11s de novo).

    Vai um contorno claro por tras de cada caixa para ela nao sumir sobre
    sombra ou solo escuro, onde o preto puro desapareceria.
    """
    img = ler_imagem(imagem)
    out = img.copy()
    esp = max(1, round(min(out.shape[:2]) / 900))
    halo = (255, 255, 255) if sum(cor) < 384 else (0, 0, 0)
    for b in caixas:
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(out, p1, p2, halo, esp + 2)
        cv2.rectangle(out, p1, p2, cor, esp)
    return out


class DetectorAgave:
    def __init__(self, pesos: str | Path = "modelo/best.pt", device: str = "cpu",
                 conf: float = CONF_PADRAO, batch: int = 4) -> None:
        """`device`: 'cpu', ou '0' para a primeira GPU.

        `batch` e quantos tiles vao juntos para a rede. 4 cabe em 6GB de VRAM.
        """
        pesos = Path(pesos)
        if not pesos.exists():
            raise FileNotFoundError(f"pesos nao encontrados: {pesos.resolve()}")
        self.modelo = YOLO(str(pesos))
        self.device = device
        self.conf = conf
        self.batch = batch
        # O FastAPI roda rota `def` (sincrona) num threadpool, entao duas
        # requisicoes podem cair aqui ao mesmo tempo. O modelo do Ultralytics
        # nao e garantido thread-safe, e a inferencia ja usa todos os nucleos —
        # paralelizar nao ganharia nada e poderia corromper a saida.
        self._trava = threading.Lock()

    _ler = staticmethod(ler_imagem)

    # ---------------------------------------------------------------- inferencia
    def _detectar_array(self, img: np.ndarray, conf: float) -> np.ndarray:
        """-> array (N, 5): [x1, y1, x2, y2, conf] nas coordenadas da imagem."""
        h, w = img.shape[:2]
        grade = list(tile_grid(w, h, TILE, OVERLAP))
        achados: list[np.ndarray] = []

        for i in range(0, len(grade), self.batch):
            lote = grade[i:i + self.batch]
            recortes = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in lote]
            saidas = self.modelo.predict(recortes, imgsz=IMGSZ, conf=conf,
                                         iou=IOU_NMS, device=self.device,
                                         verbose=False)
            for (x0, y0, x1, y1), r in zip(lote, saidas):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()[:, None]

                # Caixa encostada na borda do tile e quase sempre um PEDACO de
                # planta. A sobreposicao garante que ela aparece inteira no tile
                # vizinho, entao descartar aqui nao perde a deteccao. Bordas que
                # sao a borda da IMAGEM nao contam: ali nao ha vizinho.
                tw, th = x1 - x0, y1 - y0
                m = 2
                na_borda = (((xyxy[:, 0] <= m) & (x0 > 0))
                            | ((xyxy[:, 1] <= m) & (y0 > 0))
                            | ((xyxy[:, 2] >= tw - m) & (x1 < w))
                            | ((xyxy[:, 3] >= th - m) & (y1 < h)))
                manter = ~na_borda
                if not manter.any():
                    continue
                xyxy, cf = xyxy[manter], cf[manter]

                xyxy[:, [0, 2]] += x0            # tile -> imagem inteira
                xyxy[:, [1, 3]] += y0
                achados.append(np.hstack([xyxy, cf]))

        if not achados:
            return np.zeros((0, 5))

        det = np.vstack(achados)
        # NMS global: com sobreposicao de 40%, a mesma planta e detectada em
        # varios tiles. Sem isto a contagem sai ~3x inflada.
        idx = nms(torch.from_numpy(det[:, :4]).float(),
                  torch.from_numpy(det[:, 4]).float(), IOU_NMS).numpy()
        det = det[idx]
        return det[np.argsort(-det[:, 4])]

    # -------------------------------------------------------------------- publica
    def detectar(self, imagem: Imagem, conf: float | None = None) -> dict[str, Any]:
        """Detecta agaves e devolve um dicionario pronto para virar JSON.

        {
          "total": 678,
          "largura": 5472, "altura": 3078,
          "conf_usada": 0.3,
          "avisos": [],
          "caixas": [{"x1":.., "y1":.., "x2":.., "y2":.., "conf":..}, ...]
        }

        Coordenadas em PIXELS da imagem enviada, inteiras, canto superior
        esquerdo na origem. Ordenadas por confianca decrescente.
        """
        img = self._ler(imagem)
        h, w = img.shape[:2]
        avisos: list[str] = []

        if max(w, h) < LARGURA_MINIMA_ESPERADA:
            avisos.append(
                f"imagem de {w}x{h}: pequena demais. O modelo foi treinado em "
                f"fotos de 5472x3078, onde a planta tem ~130px. Numa imagem "
                f"redimensionada as plantas ficam pequenas demais e a deteccao "
                f"cai muito. Envie o arquivo ORIGINAL do drone."
            )

        c = self.conf if conf is None else conf
        with self._trava:                 # serializa: ver comentario no __init__
            det = self._detectar_array(img, c)
        return {
            "total": int(len(det)),
            "largura": int(w),
            "altura": int(h),
            "conf_usada": float(c),
            "avisos": avisos,
            "caixas": [
                {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                 "conf": round(float(cf), 4)}
                for x1, y1, x2, y2, cf in det
            ],
        }

    def desenhar(self, imagem: Imagem, resultado: dict[str, Any] | None = None,
                 cor: tuple[int, int, int] = PRETO) -> np.ndarray:
        """Atalho: detecta (se preciso) e desenha.

        Para a rota de DOWNLOAD, nao use este metodo — use a funcao
        `desenhar_caixas()`, que nao precisa do modelo carregado.
        """
        img = self._ler(imagem)
        if resultado is None:
            resultado = self.detectar(img)
        return desenhar_caixas(img, resultado["caixas"], cor)
