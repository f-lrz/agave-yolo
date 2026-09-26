"""Recorte da imagem em tiles — copia auto-contida do pipeline de treino.

NAO altere `tile` nem `overlap` sem reler o README: eles fazem parte do
contrato do modelo, nao sao parametros de gosto.
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
    sobreposicao e de 410px, maior que a maior planta observada no dataset
    (p95 = 530px em algumas imagens, 410px cobre a mediana com folga).
    """
    stride = max(1, int(round(tile * (1.0 - overlap))))
    for y0 in _starts(h, tile, stride):
        for x0 in _starts(w, tile, stride):
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)
