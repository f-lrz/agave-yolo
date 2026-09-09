"""Funcoes de tiling compartilhadas.

Imagens de drone tem 4000x3000+ px. O YOLO redimensiona a entrada para 640-1024,
o que faz um agave de 100px virar 20px e some da deteccao. A solucao padrao e
recortar a imagem em tiles com sobreposicao, treinar/inferir nos tiles, e
remontar o resultado nas coordenadas da imagem original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG"}


def _starts(size: int, tile: int, stride: int) -> list[int]:
    """Posicoes iniciais dos tiles em um eixo. O ultimo tile e encostado na
    borda para que nenhuma faixa da imagem fique sem cobertura."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_grid(w: int, h: int, tile: int, overlap: float = 0.2) -> Iterator[tuple[int, int, int, int]]:
    """Gera (x0, y0, x1, y1) cobrindo a imagem inteira.

    A sobreposicao existe para que uma planta cortada pela borda de um tile
    apareca inteira no tile vizinho.
    """
    stride = max(1, int(round(tile * (1.0 - overlap))))
    for y0 in _starts(h, tile, stride):
        for x0 in _starts(w, tile, stride):
            yield x0, y0, min(x0 + tile, w), min(y0 + tile, h)


def read_yolo_labels(path: Path, w: int, h: int) -> np.ndarray:
    """Le um .txt no formato YOLO (cls cx cy bw bh, normalizados).

    Retorna array (N, 5) com [cls, x1, y1, x2, y2] em pixels absolutos.
    Arquivo ausente ou vazio retorna array (0, 5).
    """
    rows: list[list[float]] = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = (float(p) for p in parts[:5])
            rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def boxes_for_tile(boxes: np.ndarray, tile_box: tuple[int, int, int, int],
                   min_visibility: float = 0.3) -> np.ndarray:
    """Recorta as caixas para dentro do tile.

    Uma caixa cortada pela borda so e mantida se ao menos `min_visibility` da
    sua area original continuar visivel. Sem esse filtro voce ensina o modelo
    que um pedaco de folha e uma planta inteira, o que dispara falsos positivos.
    """
    if len(boxes) == 0:
        return np.zeros((0, 5))
    x0, y0, x1, y1 = tile_box
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    cx1 = np.clip(boxes[:, 1], x0, x1)
    cy1 = np.clip(boxes[:, 2], y0, y1)
    cx2 = np.clip(boxes[:, 3], x0, x1)
    cy2 = np.clip(boxes[:, 4], y0, y1)
    cw = np.maximum(cx2 - cx1, 0.0)
    ch = np.maximum(cy2 - cy1, 0.0)
    vis = np.where(area > 0, (cw * ch) / np.maximum(area, 1e-9), 0.0)
    keep = (vis >= min_visibility) & (cw > 2) & (ch > 2)
    out = np.stack([boxes[:, 0], cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0], axis=1)
    return out[keep]


def to_yolo_lines(boxes: np.ndarray, w: int, h: int) -> list[str]:
    """[cls, x1, y1, x2, y2] em pixels -> linhas YOLO normalizadas."""
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix in IMG_EXT)


def label_path_for(img: Path, labels_dir: Path) -> Path:
    return labels_dir / (img.stem + ".txt")
