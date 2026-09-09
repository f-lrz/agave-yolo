"""ETAPA 1 — treina o detector de agave (1 classe).

Defaults calibrados para GTX 1660 Super 6GB:
  * amp=False  -> a serie GTX 16xx (Turing sem Tensor Cores) tem bug conhecido
                  de NaN/lentidao com mixed precision. NAO ligue.
  * workers=2  -> no Windows, muitos workers travam o dataloader.
  * batch=4 @ imgsz=1024 com yolov8s cabe em ~5GB. Se der OOM: batch=2, ou
    yolov8n, ou --imgsz 768.

    python src/train_det.py --data data/tiles/data.yaml --model yolov8s.pt --epochs 150
    python src/train_det.py --data ... --preset colab      # T4 16GB
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

PRESETS = {
    "gtx1660": dict(model="yolov8s.pt", imgsz=1024, batch=4, workers=2, amp=False, device="0"),
    "colab":   dict(model="yolov8s.pt", imgsz=1024, batch=8, workers=4, amp=True,  device="0"),
    "cpu":     dict(model="yolov8n.pt", imgsz=640,  batch=4, workers=0, amp=False, device="cpu"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="gtx1660")
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="agave_det")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in ("model", "imgsz", "batch", "device"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    model = YOLO(cfg.pop("model"))
    # sem `project=`: o Ultralytics ja resolve para runs/detect/<name>.
    # Passar project="runs/detect" duplica o caminho (runs/detect/runs/detect/...).
    model.train(
        data=args.data,
        epochs=args.epochs,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=50,
        cache=False,
        # --- augmentacao pensada para drone ---
        degrees=180.0,    # vista de cima: o agave nao tem "para cima"
        fliplr=0.5,
        flipud=0.5,       # idem — vale espelhar na vertical, ao contrario de foto normal
        scale=0.5,        # tolera variacao de altura de voo
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,        # variacao de iluminacao/hora do voo: o que mais quebra na pratica
        mosaic=1.0,
        close_mosaic=15,  # desliga o mosaic no fim, para o modelo ver a cena real
        translate=0.1,
        erasing=0.0,
        **cfg,
    )
    print(f"\nMelhor peso: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
