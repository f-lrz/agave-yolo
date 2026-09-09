"""ETAPA 2 — treina o classificador sadia/doente sobre os recortes.

Espera a estrutura criada por make_crops.py depois de voce separar as pastas:
    data/crops/train/sadia/*.jpg   data/crops/train/doente/*.jpg
    data/crops/val/sadia/*.jpg     data/crops/val/doente/*.jpg

    python src/train_cls.py --data data/crops --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta com train/ e val/")
    ap.add_argument("--model", default="yolov8s-cls.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", action="store_true", help="NAO use na GTX 16xx")
    ap.add_argument("--name", default="agave_saude")
    args = ap.parse_args()

    root = Path(args.data)
    for split in ("train", "val"):
        for cls in ("sadia", "doente"):
            n = len(list((root / split / cls).glob("*.jpg"))) if (root / split / cls).exists() else 0
            print(f"  {split}/{cls}: {n} recortes")
            if n == 0:
                raise SystemExit(f"{root/split/cls} esta vazia — separe os recortes primeiro.")

    model = YOLO(args.model)
    model.train(
        data=str(root.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, amp=args.amp, workers=2,
        name=args.name, exist_ok=True, patience=20,
        degrees=180.0, fliplr=0.5, flipud=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        erasing=0.2,
    )
    print(f"\nMelhor peso: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
