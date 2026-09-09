"""Gera imagens sinteticas de plantacao de agave, com labels YOLO.

Serve para validar a instalacao e para calibrar o cv_proposals.py antes de
gastar tempo com dados reais. Inclui de proposito as armadilhas que quebram
visao computacional classica em campo:

  * mato / capim  -> blobs verdes esparramados, que o filtro de solidez precisa
                     rejeitar (sao verdes tanto quanto o agave)
  * sombras       -> regioes escuras; um indice de vegetacao ruim perde a planta
  * plantas encostadas -> exigem o watershed para separar
  * tamanho variavel   -> nem todo agave tem o mesmo porte

    python tools/gerar_dados_falsos.py --n 9 --out data/raw
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def solo(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Solo marrom com textura de baixa frequencia + granulado fino."""
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (58 + rng.uniform(-8, 8), 82 + rng.uniform(-8, 8), 108 + rng.uniform(-10, 10))
    # GaussianBlur devolve (h, w) e nao (h, w, 1): reabre o eixo para broadcast
    grande = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0, 0), 60)
    grande = grande[:, :, None] * 260
    fino = np.random.randn(h, w, 3).astype(np.float32) * 6
    return np.clip(base + grande + fino, 0, 255).astype(np.uint8)


def agave(img: np.ndarray, cx: int, cy: int, r: int, rng: random.Random,
          vigor: float = 1.0) -> None:
    """Roseta radial. `vigor` < 1 deixa a planta mais amarelada e menor."""
    n = rng.randint(13, 20)
    esp = max(3, int(r * 0.14))
    for i in range(n):
        t = np.deg2rad(i * 360 / n + rng.uniform(-8, 8))
        comp = r * rng.uniform(0.8, 1.0)
        cor = (int(62 * vigor + rng.uniform(-10, 10)),
               int(140 * vigor + rng.uniform(-18, 18)),
               int(70 + (1 - vigor) * 70 + rng.uniform(-10, 10)))
        cv2.line(img, (cx, cy), (int(cx + comp * np.cos(t)), int(cy + comp * np.sin(t))),
                 cor, esp, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(r * 0.16)), (55, 120, 60), -1, cv2.LINE_AA)


def mato(img: np.ndarray, cx: int, cy: int, r: int, rng: random.Random) -> None:
    """Capim: verde, porem esparramado e irregular — nao deve virar caixa."""
    for _ in range(rng.randint(12, 22)):
        t = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0, r)
        x, y = int(cx + d * np.cos(t)), int(cy + d * np.sin(t))
        t2 = rng.uniform(0, 2 * np.pi)
        cv2.line(img, (x, y), (int(x + r * 0.5 * np.cos(t2)), int(y + r * 0.5 * np.sin(t2))),
                 (50 + rng.randint(-10, 10), 125 + rng.randint(-20, 20), 60 + rng.randint(-10, 10)),
                 max(2, int(r * 0.09)), cv2.LINE_AA)


def sombra(img: np.ndarray, rng: random.Random) -> None:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(1, 3)):
        cv2.ellipse(mask, (rng.randint(0, w), rng.randint(0, h)),
                    (rng.randint(w // 8, w // 3), rng.randint(h // 8, h // 3)),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 90)[:, :, None]
    img[:] = np.clip(img.astype(np.float32) * (1 - 0.42 * mask), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--largura", type=int, default=4000)
    ap.add_argument("--altura", type=int, default=3000)
    ap.add_argument("--raio", type=int, default=65, help="raio medio do agave em px")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    W, H = args.largura, args.altura

    for k in range(args.n):
        img = solo(H, W, rng)
        linhas = []
        passo_x, passo_y = W // 8, H // 6

        for gy in range(6):
            for gx in range(8):
                if rng.random() < 0.05:                    # falha no plantio
                    continue
                cx = passo_x // 2 + gx * passo_x + rng.randint(-45, 45)
                cy = passo_y // 2 + gy * passo_y + rng.randint(-45, 45)
                r = int(args.raio * rng.uniform(0.75, 1.25))
                vigor = rng.uniform(0.55, 1.0)
                agave(img, cx, cy, r, rng, vigor)
                linhas.append(f"0 {cx / W:.6f} {cy / H:.6f} {2 * r / W:.6f} {2 * r / H:.6f}")

                if rng.random() < 0.10:                    # vizinha encostada
                    t = rng.uniform(0, 2 * np.pi)
                    d = int(r * rng.uniform(1.1, 1.5))
                    cx2, cy2 = cx + int(d * np.cos(t)), cy + int(d * np.sin(t))
                    if r < cx2 < W - r and r < cy2 < H - r:
                        r2 = int(r * rng.uniform(0.8, 1.1))
                        agave(img, cx2, cy2, r2, rng, rng.uniform(0.6, 1.0))
                        linhas.append(f"0 {cx2 / W:.6f} {cy2 / H:.6f} "
                                      f"{2 * r2 / W:.6f} {2 * r2 / H:.6f}")

        for _ in range(rng.randint(18, 35)):               # mato, sem label
            mato(img, rng.randint(0, W), rng.randint(0, H),
                 int(args.raio * rng.uniform(0.4, 0.9)), rng)

        sombra(img, rng)
        cv2.imwrite(str(args.out / "images" / f"voo_{k:02d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        (args.out / "labels" / f"voo_{k:02d}.txt").write_text("\n".join(linhas))
        print(f"voo_{k:02d}.jpg  {len(linhas)} agaves")

    print(f"\n{args.n} imagens {W}x{H} em {args.out}")


if __name__ == "__main__":
    main()
