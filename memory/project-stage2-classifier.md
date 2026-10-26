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

**Iteração 2 (24/10/2026):** garimpei as [[reference-fotos-novas-voo2]] — detector `best_v5.pt` → 6245 caixas → `make_crops.py --cls` (ranqueado pelo v1, nova flag que eu adicionei) em `data/crops_voo2`. Usuário triou: **57 doentes novas** (revisou pela montagem `data/crops_voo2/_revisar_doente.jpg`, removeu as de centro sadio). Juntei voo1+voo2 em **`data/crops_v2`** (234 doentes, 20.578 sadias) via merge → `fechar_crops` → split: **train 183d/1839s, val 51d/501s**. Treinando **v2 na CPU** → `runs/classify/agave_saude_v2` (vs v1 137d/40d).

**v2 pronto (best.pt em `runs/classify/agave_saude_v2/weights/`).** O treino na CPU chegou à época 59/60 e foi morto pelo reaper de memória do Claude Code (máquina estava suspensa/ociosa ~2 dias) — best.pt ficou íntegro. Avaliado (`--val data/crops_v2/val`, 51 doentes): **AUC 0,960** (v1 0,934); recall doente @0,50 **76,5%** (v1 62,5%) com precisão **81,2%** mantida; garimpeiro top 20% = 49/51 (96%). Ganho real dos dados novos. Ressalva: val do v1 e do v2 são conjuntos diferentes.

**Decisão de produto (26/10/2026):** uso = **triagem pro agrônomo** (ranquear suspeitas, o especialista confirma; não cravar limiar). Não há mais fotos novas.

**PENDENTE — revisão do agrônomo:** como não há foto nova, a alavanca é melhor rótulo. Rodei o v2 sobre as 20.578 sadias atuais e separei os **300 maiores suspeitos** (doentes que a 1ª triagem, feita por não-especialista, pode ter perdido) em `data/crops_v2/revisao_agronomo/_revisar` + folhas de contato `contato_folha1..3.jpg` + `index_revisao.csv`. O agrônomo arrasta as confirmadas doentes para `data/crops_v2/revisao_agronomo/doente`. Quando ele terminar: incorporar essas doentes (rótulo de especialista) ao dataset, retreinar **v3** (usuário roda `train_cls.py` no próprio terminal — não em background, por causa do reaper de memória), e medir. Scripts no scratchpad da sessão: `score_para_agronomo.py`, `montagem_lote.py`, `montagem_doente.py`.

**(Histórico) Alternativa de garimpo:** com 234 doentes (alvo README 300), outra rodada com o v2 (AUC 0,960) exigiria fotos novas, que não há. Tweaks de modelo (yolov8m-cls etc.) são secundários. **Atenção ao treinar de novo:** rodar na CPU leva ~30min de compute, mas se a máquina dormir o processo em background pode ser morto por falta de memória — melhor o usuário rodar `train_cls.py` num terminal próprio, ou setar `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`. Mudança não commitada em `src/make_crops.py`: flag `--cls`/`--cls-device` (ranqueia pela prob do classificador em vez da cor). Scripts de apoio ainda não commitados: `tools/avaliar_cls.py`, `tools/aplicar_triagem.py`, `tools/conf_doente.py`.
