"""Confere se o ambiente tem tudo instalado, e diz o comando exato do que falta.

Separa em dois grupos, porque eles nao sao necessarios ao mesmo tempo:

  ESTAGIO 0 (cv_proposals, eval_proposals, checar_dados) -> numpy, OpenCV, tqdm.
      Nao precisa de PyTorch nem de GPU. Da para comecar a rotular hoje.
  ESTAGIO 1 (treinar o YOLO) -> torch, torchvision, ultralytics, e de preferencia
      uma GPU.

    python tools/checar_ambiente.py
"""

from __future__ import annotations

import importlib
import platform
import sys

OK, AVISO, ERRO = "  ok  ", "  !!  ", "  XX  "

# (modulo importavel, nome no pip, versao minima recomendada)
ESTAGIO0 = [("numpy", "numpy", (1, 24)),
            ("cv2", "opencv-python", (4, 8)),
            ("tqdm", "tqdm", (4, 60))]
ESTAGIO1 = [("torch", "torch", (2, 0)),
            ("torchvision", "torchvision", (0, 15)),
            ("ultralytics", "ultralytics", (8, 3))]


def versao(mod) -> str:
    return getattr(mod, "__version__", "?")


def como_tupla(v: str) -> tuple[int, ...]:
    partes = []
    for p in v.split(".")[:3]:
        num = "".join(c for c in p if c.isdigit())
        partes.append(int(num) if num else 0)
    return tuple(partes)


def checar(grupo, titulo: str) -> list[str]:
    print(f"\n== {titulo} ==")
    faltando = []
    for modname, pipname, minimo in grupo:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            print(ERRO + f"{pipname:<18} NAO INSTALADO")
            faltando.append(pipname)
            continue
        v = versao(mod)
        if minimo and como_tupla(v) < minimo:
            alvo = ".".join(str(x) for x in minimo)
            print(AVISO + f"{pipname:<18} {v}  (recomendado >= {alvo})")
        else:
            print(OK + f"{pipname:<18} {v}")
    return faltando


def checar_gpu() -> str:
    """-> 'ok', 'cpu_build', 'sem_cuda' ou 'sem_torch'."""
    print("\n== GPU ==")
    try:
        import torch
    except ImportError:
        print(AVISO + "torch nao instalado — nada a checar por enquanto")
        return "sem_torch"

    # Pegadinha do Windows: `pip install torch` sem o index-url instala a build
    # SO DE CPU, sem avisar. Ela importa, roda, e treina 30x mais devagar.
    if torch.version.cuda is None:
        print(ERRO + "esta e a build de CPU do PyTorch (torch.version.cuda = None)")
        print("       Sua GPU nao sera usada. Reinstale com a build CUDA:")
        print("         pip uninstall -y torch torchvision")
        print("         pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu121")
        return "cpu_build"

    print(OK + f"PyTorch compilado com CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print(ERRO + "CUDA nao disponivel em tempo de execucao")
        print("       Driver NVIDIA desatualizado ou ausente. Rode `nvidia-smi`")
        print("       num terminal: se falhar, o problema e o driver.")
        return "sem_cuda"

    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1024 ** 3
    print(OK + f"{p.name}  |  {vram:.1f} GB  |  compute capability {p.major}.{p.minor}")

    if p.major == 7 and p.minor == 5 and "16" in p.name:
        print(AVISO + "GTX 16xx: Turing SEM Tensor Cores. Mixed precision (AMP) da")
        print("       NaN no loss ou fica mais lenta que FP32. Os scripts daqui ja")
        print("       usam amp=False no preset gtx1660 — nao mude isso.")
    if vram < 7:
        print(AVISO + f"{vram:.0f} GB de VRAM: use --preset gtx1660 (batch 4 @ imgsz 1024).")
        print("       Se der CUDA out of memory: --batch 2, ou --model yolov8n.pt.")
    return "ok"


def main() -> None:
    print(f"\nPython {platform.python_version()}  ({platform.system()} "
          f"{platform.release()})")
    if sys.version_info < (3, 9):
        print(ERRO + "Python muito antigo. O Ultralytics pede 3.8+, e 3.10-3.12 "
                     "sao os mais bem suportados.")
    elif sys.version_info >= (3, 13):
        print(AVISO + "Python muito novo — algumas bibliotecas ainda nao publicam "
                      "wheels. Se a instalacao falhar, use Python 3.11 ou 3.12.")

    falta0 = checar(ESTAGIO0, "ESTAGIO 0 — rotular sem treinar (precisa agora)")
    falta1 = checar(ESTAGIO1, "ESTAGIO 1 — treinar o YOLO (precisa depois)")
    gpu = checar_gpu()

    print("\n== O que fazer ==")
    if falta0:
        print(ERRO + "Falta o basico. Instale antes de qualquer coisa:")
        print(f"         pip install {' '.join(falta0)}")
    else:
        print(OK + "Estagio 0 PRONTO. Pode seguir o passo a passo do README hoje,")
        print("       mesmo que a parte de GPU abaixo esteja pendente.")

    if falta1:
        print(AVISO + "Estagio 1 incompleto (so precisa quando for treinar):")
        if "torch" in falta1 or "torchvision" in falta1:
            print("         pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu121")
        resto = [x for x in falta1 if x not in ("torch", "torchvision")]
        if resto:
            print(f"         pip install {' '.join(resto)}")
    elif gpu == "cpu_build":
        print(AVISO + "Estagio 1: pacotes ok, mas o PyTorch e build de CPU. Treinar")
        print("       assim funciona e leva ~30x mais tempo. Reinstale antes de treinar")
        print("       (comando na secao GPU acima).")
    elif gpu == "sem_cuda":
        print(AVISO + "Estagio 1: pacotes ok, mas a GPU nao esta acessivel. Resolva o")
        print("       driver, ou treine no Colab com --preset colab.")
    else:
        print(OK + "Estagio 1 pronto tambem.")


if __name__ == "__main__":
    main()
