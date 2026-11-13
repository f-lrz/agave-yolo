"""Servico de visao computacional: detecta agaves na foto de drone e faz a
TRIAGEM de saude de cada planta.

Duas etapas, dois modelos:
  1. detector  (models/best.pt)               onde estao as agaves
  2. saude     (models/classificador_saude.pt) prob_doente de cada agave

O QUE A SAIDA DE SAUDE E: uma triagem. `prob_doente` ordena as plantas por
suspeita para o agronomo conferir em campo. NAO e laudo nem diagnostico — ver
models/MODELO_SAUDE.md antes de escrever qualquer texto para o usuario.

Variaveis de ambiente (todas opcionais):
  AGAVE_MOCK=1             sem modelos (frontend sem torch): resposta simulada
  AGAVE_PESOS              detector              (models/best.pt)
  AGAVE_PESOS_SAUDE        classificador de saude (models/classificador_saude.pt)
  AGAVE_SAUDE=0            desliga a etapa 2: volta a ser so deteccao
  AGAVE_LIMIAR_DOENTE      prob_doente a partir da qual conta como doente (0.5)
  AGAVE_DEVICE             'cpu' ou '0' para a primeira GPU
  AGAVE_CONF               limiar do detector (0.30, calibrado para contagem)
  AGAVE_JPEG               qualidade do JPEG devolvido (85)
  AGAVE_COR                cor das caixas sadias (ciano)
  AGAVE_COR_DOENTE         cor das caixas doentes (vermelho)
  AGAVE_MOSTRAR_SADIAS=0   desenha so as doentes
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

USAR_MOCK = os.getenv("AGAVE_MOCK", "0") == "1"
PESOS = Path(os.getenv("AGAVE_PESOS", "models/best.pt"))
PESOS_SAUDE = Path(os.getenv("AGAVE_PESOS_SAUDE", "models/classificador_saude.pt"))
USAR_SAUDE = os.getenv("AGAVE_SAUDE", "1") == "1"
LIMIAR_DOENTE = float(os.getenv("AGAVE_LIMIAR_DOENTE", "0.5"))
DEVICE = os.getenv("AGAVE_DEVICE", "cpu")              # '0' para a primeira GPU
CONF = float(os.getenv("AGAVE_CONF", "0.30"))          # calibrado para contagem
QUALIDADE_JPEG = int(os.getenv("AGAVE_JPEG", "85"))

# Cores: preto, vermelho, magenta, ciano, amarelo, branco, verde. Cada caixa
# leva um contorno claro por tras, para nao sumir sobre sombra. Evite 'verde'
# (confunde com a planta). O vermelho fica reservado para as doentes.
NOME_COR = os.getenv("AGAVE_COR", "ciano")
NOME_COR_DOENTE = os.getenv("AGAVE_COR_DOENTE", "vermelho")
MOSTRAR_SADIAS = os.getenv("AGAVE_MOSTRAR_SADIAS", "1") == "1"

_detector = None
_classificador = None
_trava_carga = threading.Lock()


def obter_detector():
    """Carrega o detector uma unica vez.

    O import fica aqui dentro de proposito: com AGAVE_MOCK=1 o modulo nao
    precisa de torch nem ultralytics instalados.
    """
    global _detector
    if _detector is None:
        with _trava_carga:
            if _detector is None:                      # outra thread pode ter carregado
                from .agave_detector import DetectorAgave
                _detector = DetectorAgave(PESOS, device=DEVICE, conf=CONF)
    return _detector


def obter_classificador():
    """Carrega o classificador de saude uma unica vez (None se AGAVE_SAUDE=0)."""
    global _classificador
    if not USAR_SAUDE:
        return None
    if _classificador is None:
        with _trava_carga:
            if _classificador is None:
                from .agave_detector import ClassificadorSaude
                _classificador = ClassificadorSaude(PESOS_SAUDE, device=DEVICE)
    return _classificador


def obter_modelos() -> None:
    """Carrega os dois modelos. Chame no startup (lifespan) do app."""
    obter_detector()
    obter_classificador()


def _codificar(img: np.ndarray) -> str:
    sucesso, buffer = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
    if not sucesso:
        raise ValueError("Erro ao recodificar a imagem.")
    return base64.b64encode(buffer).decode("utf-8")


def _analisar_sincrono(file_bytes: bytes) -> dict[str, Any]:
    """Roda a inferencia. BLOQUEANTE — sempre chame via asyncio.to_thread."""
    from .agave_detector import CORES, CIANO, VERMELHO, analisar, desenhar_saude, ler_imagem

    img = ler_imagem(file_bytes)               # decodifica UMA vez (~50 MB)
    r = analisar(obter_detector(), obter_classificador(), img, limiar=LIMIAR_DOENTE)
    anotada = desenhar_saude(img, r["plantas"], limiar=LIMIAR_DOENTE,
                             cor_sadia=CORES.get(NOME_COR, CIANO),
                             cor_doente=CORES.get(NOME_COR_DOENTE, VERMELHO),
                             mostrar_sadias=MOSTRAR_SADIAS)
    return {
        "image_base64": _codificar(anotada),
        "total_agaves": r["total"],
        "largura": r["largura"],
        "altura": r["altura"],
        "conf": r["conf_usada"],
        "avisos": r["avisos"],
        "saude_ativa": r["saude_ativa"],
        "limiar_doente": r["limiar_doente"],
        "doentes": r["doentes"],
        "sadias": r["sadias"],
        "taxa_doenca": r["taxa_doenca"],
        "plantas": r["plantas"],
    }


async def process_image(file_bytes: bytes) -> dict[str, Any]:
    """Detecta + triagem de saude; devolve a imagem marcada em base64 e o laudo.

    A inferencia leva ~12s em CPU (~11s deteccao + ~1s saude) e e 100%
    bloqueante (numpy/torch nao liberam o event loop). Rodar isso direto num
    `async def` travaria o servidor INTEIRO durante esse tempo — inclusive o
    /health e as outras requisicoes. O `to_thread` joga o trabalho para um
    worker e mantem o loop livre.
    """
    if USAR_MOCK:
        return await _process_image_mock(file_bytes)
    return await asyncio.to_thread(_analisar_sincrono, file_bytes)


async def _process_image_mock(file_bytes: bytes) -> dict[str, Any]:
    """Resposta simulada, para desenvolvimento de frontend sem os modelos.

    Devolve UMA planta sadia e UMA doente, para a tela de triagem ter o que
    mostrar. Mesmo formato da resposta real.
    """
    await asyncio.sleep(3)

    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Erro ao decodificar a imagem.")

    altura, largura, _ = img.shape
    s = max(40, min(largura, altura) // 12)
    plantas = [
        {"x1": largura // 3, "y1": altura // 3, "x2": largura // 3 + s,
         "y2": altura // 3 + s, "conf": 0.9, "prob_doente": 0.93, "saude": "doente"},
        {"x1": largura // 2, "y1": altura // 2, "x2": largura // 2 + s,
         "y2": altura // 2 + s, "conf": 0.9, "prob_doente": 0.04, "saude": "sadia"},
    ]
    for p in plantas:
        cor = (0, 0, 255) if p["saude"] == "doente" else (255, 255, 0)
        cv2.rectangle(img, (p["x1"], p["y1"]), (p["x2"], p["y2"]), cor, 3)
    cv2.putText(img, "MOCK - sem modelo carregado", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return {"image_base64": _codificar(img), "total_agaves": 2,
            "largura": largura, "altura": altura, "conf": 0.0,
            "avisos": ["MODO MOCK: nenhuma deteccao real foi feita."],
            "saude_ativa": True, "limiar_doente": LIMIAR_DOENTE,
            "doentes": 1, "sadias": 1, "taxa_doenca": 0.5, "plantas": plantas}
