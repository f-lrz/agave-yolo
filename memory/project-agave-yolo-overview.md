---
name: project-agave-yolo-overview
description: Visão geral do projeto agave-yolo (Identify 2.0) — pipeline 2 estágios de detecção de agave doente
metadata:
  type: project
---

Projeto **agave-yolo** (parte do produto "Identify 2.0", backend `identify2.0-backend` em FastAPI). Detecta **agaves doentes** em fotos aéreas de drone (5472x3078). Pipeline de **2 estágios**, decisão deliberada porque doente é minoria (<1%): num detector único de 2 classes a classe rara vira caixa perdida.

- **Estágio 1 — detector** (`YOLOv8s`, 1 classe `agave`): **COMPLETO**. Pesos `best_v5.pt` (iterações best.pt..best_v5.pt), 42 fotos / 14.568 caixas, mAP50=0,867, erro de contagem ~4,9% em conf=0,30. Já empacotado para o backend em `entrega_detector/` (lib + `MODELO.md` + `teste_manual/`). Usa tiling 1024 + NMS global (YOLO direto na foto inteira perde metade das plantas).
- **Estágio 2 — classificador** (`YOLOv8s-cls`, sadia|doente): em progresso. Ver [[project-stage2-classifier]].

Repositório é **muito bem documentado** (README raiz enorme, docstrings longas em cada script). Convenção de classes: `0=doente`, `1=sadia`. Há uma GTX 1660 SUPER 6GB, mas ela **tem defeito e o usuário prefere CPU** (ver [[project-gpu-com-defeito]]). venv em `./venv` (python 3.13).
