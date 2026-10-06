"""Detector de agave em imagens de drone — apenas DETECCAO, sem diagnostico."""

from .detector import (AMARELO, BRANCO, CIANO, CONF_PADRAO, CONF_PRE_ANOTACAO,
                       CORES, IMGSZ, MAGENTA, OVERLAP, PRETO, TILE, VERDE,
                       VERMELHO, DetectorAgave, desenhar_caixas, ler_imagem)

__all__ = ["DetectorAgave", "desenhar_caixas", "ler_imagem",
           "CONF_PADRAO", "CONF_PRE_ANOTACAO", "TILE", "OVERLAP", "IMGSZ",
           "CORES", "PRETO", "VERMELHO", "MAGENTA", "CIANO", "AMARELO",
           "BRANCO", "VERDE"]
__version__ = "1.2.0"
