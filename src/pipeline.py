"""Pipeline final de 2 estagios: detecta agaves -> classifica sadia/doente.

E este arquivo que o seu backend deve importar. Carrega os dois modelos uma vez
e expoe `AgavePipeline.run(imagem)` devolvendo JSON + imagem anotada.

    # linha de comando
    python src/pipeline.py --det runs/detect/agave/weights/best.pt \
        --cls runs/classify/saude/weights/best.pt \
        --images data/raw/novas --out out/laudo

    # no backend
    from pipeline import AgavePipeline
    pipe = AgavePipeline("det.pt", "cls.pt")     # uma vez, no startup
    laudo = pipe.run(img_bgr)                    # por requisicao
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict_tiled import draw, predict_image
from tiling import list_images

CORES = {"sadia": (0, 200, 0), "doente": (0, 0, 255)}


class AgavePipeline:
    def __init__(self, det_weights: str, cls_weights: str | None = None,
                 tile: int = 1024, overlap: float = 0.2, imgsz: int = 1024,
                 conf: float = 0.25, iou: float = 0.5, batch: int = 4,
                 device: str = "0", pad: float = 0.15, cls_size: int = 224):
        self.det = YOLO(det_weights)
        self.cls = YOLO(cls_weights) if cls_weights else None
        self.tile, self.overlap, self.imgsz = tile, overlap, imgsz
        self.conf, self.iou, self.batch = conf, iou, batch
        self.device, self.pad, self.cls_size = device, pad, cls_size

    def _crops(self, img: np.ndarray, det: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        crops = []
        for _, x1, y1, x2, y2, _ in det:
            mx, my = (x2 - x1) * self.pad, (y2 - y1) * self.pad
            c = img[max(int(y1 - my), 0):min(int(y2 + my), h),
                    max(int(x1 - mx), 0):min(int(x2 + mx), w)]
            if c.size == 0:
                c = np.zeros((self.cls_size, self.cls_size, 3), np.uint8)
            crops.append(cv2.resize(c, (self.cls_size, self.cls_size),
                                    interpolation=cv2.INTER_AREA))
        return crops

    def run(self, img: np.ndarray) -> dict:
        det = predict_image(self.det, img, self.tile, self.overlap, self.conf,
                            self.iou, self.batch, self.imgsz, self.device)

        plantas = [{"bbox": [round(float(v), 1) for v in d[1:5]],
                    "conf_deteccao": round(float(d[5]), 3),
                    "saude": None, "conf_saude": None} for d in det]

        if self.cls is not None and plantas:
            crops = self._crops(img, det)
            for i in range(0, len(crops), 32):
                for j, r in enumerate(self.cls.predict(crops[i:i + 32], device=self.device,
                                                       imgsz=self.cls_size, verbose=False)):
                    k = int(r.probs.top1)
                    plantas[i + j]["saude"] = r.names[k]
                    plantas[i + j]["conf_saude"] = round(float(r.probs.top1conf), 3)

        doentes = sum(1 for p in plantas if p["saude"] == "doente")
        return {
            "total_agaves": len(plantas),
            "doentes": doentes,
            "sadias": len(plantas) - doentes if self.cls else None,
            "taxa_doenca": round(doentes / len(plantas), 4) if plantas and self.cls else None,
            "plantas": plantas,
        }

    def annotate(self, img: np.ndarray, laudo: dict) -> np.ndarray:
        out = img.copy()
        thick = max(1, round(min(img.shape[:2]) / 900))
        for p in laudo["plantas"]:
            x1, y1, x2, y2 = (int(v) for v in p["bbox"])
            saude = p["saude"] or "agave"
            color = CORES.get(saude, (255, 160, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            cv2.putText(out, saude, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color, thick)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", default=None)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    pipe = AgavePipeline(args.det, args.cls, tile=args.tile, imgsz=args.imgsz,
                         conf=args.conf, batch=args.batch, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    for p in list_images(args.images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        laudo = pipe.run(img)
        (args.out / f"{p.stem}.json").write_text(
            json.dumps(laudo, ensure_ascii=False, indent=2), encoding="utf-8")
        cv2.imwrite(str(args.out / f"{p.stem}.jpg"), pipe.annotate(img, laudo),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{p.name}: {laudo['total_agaves']} agaves, "
              f"{laudo['doentes']} doentes ({laudo['taxa_doenca']})")


if __name__ == "__main__":
    main()
