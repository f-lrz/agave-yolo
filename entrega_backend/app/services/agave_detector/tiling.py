"""Recorte da imagem em tiles — mesma grade usada no treino do detector.

NAO altere `tile` nem `overlap` (definidos em detector.py): fazem parte do
contrato do modelo (ver models/MODELO.md), nao sao parametros de gosto.
"""

from __future__ import annotations

from typing import Iterator


def _starts(size: int, tile: int, stride: int) -> list[int]:
    """Posicoes iniciais dos tiles em um eixo.

    O ultimo tile e encostado na borda para que nenhuma faixa da imagem fique
    sem cobertura.
    """
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_grid(w: int, h: int, tile: int, overlap: float) -> Iterator[tuple[int, int, int, int]]:
    """Gera (x0, y0, x1, y1) cobrindo a imagem inteira.

    A sobreposicao existe para que uma planta cortada pela borda de um tile
    apareca inteira no tile vizinho. Com tile=1024 e overlap=0.4 a faixa de
    sobreposicao e de 410px, folgada para a planta tipica do treino (~110 a
    ~320px). Planta maior que isso (foto tirada baixo demais) pode nao caber
    inteira em nenhum tile — e por isso que o detector avisa nesse caso.
    """
    stride = max(1, int(round(tile * (1.0 - overlap))))
    for y0 in _starts(h, tile, stride):
        for x0 in _starts(w, tile, stride):
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)
