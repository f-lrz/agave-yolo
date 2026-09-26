"""Detector de agave em imagens de drone — apenas DETECCAO, sem diagnostico."""

from .detector import (CIANO, CONF_PADRAO, CONF_PRE_ANOTACAO, IMGSZ, MAGENTA,
                       OVERLAP, PRETO, TILE, DetectorAgave, desenhar_caixas,
                       ler_imagem)

__all__ = ["DetectorAgave", "desenhar_caixas", "ler_imagem",
           "CONF_PADRAO", "CONF_PRE_ANOTACAO", "TILE", "OVERLAP", "IMGSZ",
           "PRETO", "MAGENTA", "CIANO"]
__version__ = "1.1.0"
