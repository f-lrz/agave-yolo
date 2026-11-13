"""Agave em imagem de drone: deteccao (DetectorAgave) + triagem de saude
(ClassificadorSaude). Ver models/MODELO.md e models/MODELO_SAUDE.md."""

from .detector import (AMARELO, BRANCO, CIANO, CONF_PADRAO, CONF_PRE_ANOTACAO,
                       CORES, IMGSZ, MAGENTA, OVERLAP, PRETO, TILE, VERDE,
                       VERMELHO, DetectorAgave, desenhar_caixas, ler_imagem)
from .saude import (LIMIAR_DOENTE, ClassificadorSaude, analisar,
                    desenhar_saude, recortar)

__all__ = ["DetectorAgave", "desenhar_caixas", "ler_imagem",
           "ClassificadorSaude", "analisar", "desenhar_saude", "recortar",
           "LIMIAR_DOENTE", "CONF_PADRAO", "CONF_PRE_ANOTACAO", "TILE",
           "OVERLAP", "IMGSZ", "CORES", "PRETO", "VERMELHO", "MAGENTA",
           "CIANO", "AMARELO", "BRANCO", "VERDE"]
__version__ = "2.0.0"
