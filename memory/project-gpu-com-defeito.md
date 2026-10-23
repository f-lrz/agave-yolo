---
name: project-gpu-com-defeito
description: A GPU do usuário tem defeito (desliga sozinha) — preferir CPU nos treinos/inferências
metadata:
  type: project
---

A GTX 1660 SUPER da máquina do usuário tem um **defeito de hardware: desliga sozinha**. Por isso, no [[project-agave-yolo-overview]] o usuário **prefere usar CPU** (`--device cpu`), mesmo sendo mais lento. Informado em 23/10/2026.

**How to apply:** por padrão, rodar treino e inferência na CPU (`--device cpu` / `--cls-device cpu`). O classificador 224x224 treina em CPU em ~30min (foi como o v1 saiu, intencionalmente). Não recomendar `--device 0` sem o usuário pedir. Ver [[project-stage2-classifier]].
