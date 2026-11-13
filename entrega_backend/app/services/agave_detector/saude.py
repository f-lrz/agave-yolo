"""Classificador de saude: para cada agave detectado, a probabilidade de doenca.

O QUE ELE FAZ: recebe as caixas do DetectorAgave e devolve, para cada planta,
`prob_doente` (0 a 1). E uma TRIAGEM: ordena as plantas por suspeita para o
agronomo conferir em campo. Nao e laudo.

    from agave_detector import DetectorAgave, ClassificadorSaude, analisar

    det = DetectorAgave("models/best.pt")                     # UMA vez, no startup
    cls = ClassificadorSaude("models/classificador_saude.pt")  # UMA vez, no startup

    r = analisar(det, cls, bytes_do_upload)
    r["total"], r["doentes"]          # 139, 1
    r["plantas"][0]                   # a MAIS suspeita (lista ordenada)
    # {"x1":.., "y1":.., "x2":.., "y2":.., "conf": 0.81,
    #  "prob_doente": 0.97, "saude": "doente"}

Por que uma probabilidade e nao so "doente/sadia": o uso e triagem. A planta
com 0.45 ("sadia" pelo corte de 0.5) e justamente a que o agronomo mais precisa
olhar. A lista vem ordenada por prob_doente para virar a fila de revisao.

"Doente" aqui e o que foi rotulado como doente no treino. Planta SECA (morta,
ressecada, cinza) foi ensinada como NAO doente — ver models/MODELO_SAUDE.md.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .detector import (BRANCO, CIANO, PRETO, VERMELHO, DetectorAgave, Imagem,
                       desenhar_caixas, ler_imagem)

# Contrato do modelo de saude. O recorte TEM que ser feito igual ao do treino
# (agave-yolo/src/make_crops.py): caixa + 15% de margem de cada lado, cortado
# em pixels inteiros e reduzido para 224x224 com INTER_AREA. Mudar qualquer um
# muda o que o modelo ve e degrada o resultado sem dar erro.
PAD = 0.15
TAMANHO = 224
LOTE = 32

# Corte padrao para chamar a planta de "doente". So afeta o rotulo `saude` e a
# contagem `doentes`; o `prob_doente` sai sempre. Ver MODELO_SAUDE.md para a
# tabela de cortes (0.3 acha mais doentes; 0.7 da menos alarme falso).
LIMIAR_DOENTE = 0.5


def recortar(img: np.ndarray, caixa: dict[str, Any], pad: float = PAD,
             tamanho: int = TAMANHO) -> np.ndarray:
    """Recorte de UMA planta, identico ao usado para gerar o dataset de treino."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = caixa["x1"], caixa["y1"], caixa["x2"], caixa["y2"]
    mx, my = (x2 - x1) * pad, (y2 - y1) * pad
    cx1, cy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
    cx2, cy2 = min(int(x2 + mx), w), min(int(y2 + my), h)
    c = img[cy1:cy2, cx1:cx2]
    if c.size == 0:
        c = np.zeros((tamanho, tamanho, 3), np.uint8)
    return cv2.resize(c, (tamanho, tamanho), interpolation=cv2.INTER_AREA)


class ClassificadorSaude:
    def __init__(self, pesos: str | Path = "models/classificador_saude.pt",
                 device: str = "cpu", lote: int = LOTE) -> None:
        """`device`: 'cpu', ou '0' para a primeira GPU."""
        pesos = Path(pesos)
        if not pesos.exists():
            raise FileNotFoundError(f"pesos nao encontrados: {pesos.resolve()}")
        self.modelo = YOLO(str(pesos))
        nomes = self.modelo.names
        if "doente" not in nomes.values():
            raise ValueError(f"{pesos} nao e o classificador de saude "
                             f"(classes {nomes}, esperava 'doente' e 'sadia')")
        self._idx_doente = next(i for i, n in nomes.items() if n == "doente")
        self.device = device
        self.lote = lote
        # Mesmo motivo do DetectorAgave: o FastAPI pode chamar de duas threads.
        self._trava = threading.Lock()

    def prob_doente(self, imagem: Imagem, caixas: list[dict[str, Any]]) -> list[float]:
        """prob_doente de cada caixa, NA MESMA ORDEM de `caixas`."""
        if not caixas:
            return []
        img = ler_imagem(imagem)
        recortes = [recortar(img, c) for c in caixas]
        probs: list[float] = []
        with self._trava:
            for i in range(0, len(recortes), self.lote):
                saidas = self.modelo.predict(recortes[i:i + self.lote],
                                             imgsz=TAMANHO, device=self.device,
                                             verbose=False)
                probs += [round(float(r.probs.data[self._idx_doente]), 4)
                          for r in saidas]
        return probs


def analisar(detector: DetectorAgave, classificador: ClassificadorSaude | None,
             imagem: Imagem, conf: float | None = None,
             limiar: float = LIMIAR_DOENTE) -> dict[str, Any]:
    """Detecta + classifica. Devolve um dicionario pronto para virar JSON.

    {
      "total": 139, "largura": 5472, "altura": 3078,
      "conf_usada": 0.3, "avisos": [],
      "saude_ativa": true, "limiar_doente": 0.5,
      "doentes": 1, "sadias": 138, "taxa_doenca": 0.0072,
      "plantas": [ {"x1","y1","x2","y2","conf","prob_doente","saude"}, ... ]
    }

    `plantas` vem ORDENADA por prob_doente decrescente: a primeira e a mais
    suspeita. Com `classificador=None` vira so deteccao (prob_doente/saude
    nulos, doentes/sadias/taxa nulos), ordenada por confianca.

    As plantas tem x1..y2, entao servem direto para `desenhar_caixas()` e
    `desenhar_saude()`.
    """
    img = ler_imagem(imagem)
    r = detector.detectar(img, conf=conf)
    caixas = r.pop("caixas")

    if classificador is None:
        plantas = [{**c, "prob_doente": None, "saude": None} for c in caixas]
        return {**r, "saude_ativa": False, "limiar_doente": None,
                "doentes": None, "sadias": None, "taxa_doenca": None,
                "plantas": plantas}

    probs = classificador.prob_doente(img, caixas)
    plantas = [{**c, "prob_doente": p,
                "saude": "doente" if p >= limiar else "sadia"}
               for c, p in zip(caixas, probs)]
    plantas.sort(key=lambda p: -p["prob_doente"])
    doentes = sum(1 for p in plantas if p["saude"] == "doente")
    return {**r, "saude_ativa": True, "limiar_doente": limiar,
            "doentes": doentes, "sadias": len(plantas) - doentes,
            "taxa_doenca": round(doentes / len(plantas), 4) if plantas else 0.0,
            "plantas": plantas}


def desenhar_saude(imagem: Imagem, plantas: list[dict[str, Any]],
                   limiar: float = LIMIAR_DOENTE,
                   cor_sadia: tuple[int, int, int] = CIANO,
                   cor_doente: tuple[int, int, int] = VERMELHO,
                   mostrar_sadias: bool = True) -> np.ndarray:
    """Desenha as plantas: sadias finas, doentes grossas com o prob_doente escrito.

    **Nao carrega modelo.** Como `desenhar_caixas()`, serve para a rota de
    download: guarde as `plantas` do analisar() e redesenhe a partir delas.

    O `limiar` aqui so decide o que fica vermelho; pode ser diferente do usado
    no analisar() (ex.: a tela deixa o agronomo baixar para 0.3 e ver mais).
    Planta sem prob_doente (saude desligada) e desenhada como sadia.
    """
    img = ler_imagem(imagem)
    def eh_doente(p: dict[str, Any]) -> bool:
        return p.get("prob_doente") is not None and p["prob_doente"] >= limiar

    doentes = [p for p in plantas if eh_doente(p)]
    outras = [p for p in plantas if not eh_doente(p)] if mostrar_sadias else []

    out = desenhar_caixas(img, outras, cor_sadia) if outras else img.copy()
    esp = max(2, round(min(out.shape[:2]) / 600))
    escala = 0.45 * esp
    for p in doentes:
        p1, p2 = (int(p["x1"]), int(p["y1"])), (int(p["x2"]), int(p["y2"]))
        cv2.rectangle(out, p1, p2, BRANCO, esp + 2)
        cv2.rectangle(out, p1, p2, cor_doente, esp)
        texto = f"{p['prob_doente']:.2f}"
        org = (p1[0], max(p1[1] - 6, 16))
        cv2.putText(out, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, PRETO,
                    esp + 2, cv2.LINE_AA)
        cv2.putText(out, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor_doente,
                    esp, cv2.LINE_AA)
    return out
