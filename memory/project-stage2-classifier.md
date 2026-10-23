---
name: project-stage2-classifier
description: Estágio 2 do agave-yolo — estado, métricas e plano do classificador de saúde
metadata:
  type: project
---

Estágio 2 do [[project-agave-yolo-overview]]: classificador `YOLOv8s-cls` sadia|doente sobre recortes 224x224. **Meta do usuário (definida em 23/10/2026): o melhor classificador possível.**

**Primeira iteração completa (16/10/2026):** `make_crops.py` (14.567 recortes ranqueados por suspeita de cor a* do Lab no miolo) → triagem manual no Explorer arrastando só as doentes → `aplicar_triagem.py` escreveu de volta em `data/raw/labels` (backup `labels_backup_20261016_194344`) → `fechar_crops.py` (subamostra 10 sadias/doente, metade negativos difíceis) → `train_cls.py` → `runs/classify/agave_saude_v1`.

**Dataset atual:** 177 doentes no total (train 137 / val 40), ~1770 sadias subamostradas de 14.390. v1 treinou na **CPU** (~30min) — intencional, a GPU tem defeito (ver [[project-gpu-com-defeito]]); próximos treinos também na CPU.

**Desempenho v1** (via `tools/avaliar_cls.py`): AUC **0,934** (supera score de cor 0,889). Recall doente trava em ~62-65% nos limiares usáveis (0,625 @0,5 com prec 0,806). Ótimo como **garimpeiro** (top 20% = 35/40 doentes). Gargalo é **falta de dados de doente** (alvo README ≥300), não ajuste.

**Plano para melhorar:** (1) garimpar as [[reference-fotos-novas-voo2]] usando o v1 como ranqueador → triar topo → +doentes; (2) retreinar na GPU → v2; (3) iterar até recall platôar. Scripts de apoio (não commitados): `tools/avaliar_cls.py`, `tools/aplicar_triagem.py`, `tools/conf_doente.py` (checa se o detector é menos confiante nas doentes — viés de seleção).
