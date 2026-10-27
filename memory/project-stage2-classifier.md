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

**Revisão feita (26/10/2026):** rodei o v2 sobre as 20.578 sadias → 300 maiores suspeitos em `data/crops_v2/revisao_agronomo/` (folhas `contato_folha1..3.jpg` + `index_revisao.csv`). Usuário confirmou **28 doentes** que a 1ª passada perdeu → incorporadas (movidas de `_classificar` p/ `doente` no pool). Total agora **262 doentes**.

**v3 treinado e avaliado (26/10/2026):** `runs/classify/agave_saude_v3` (best.pt = época 17; parou na 37 por early stopping `patience=20`, normal). Na val limpa dele (52d/482s): **AUC 0,940**, recall @0,50 **75,0%**, precisão **88,6%** (a melhor das três), garimpeiro top 20% = 47/52. AUC abaixo do v2 (0,960) PORQUE a val do v3 é mais dura (os 28 casos difíceis recém-confirmados caíram parte nela) — não é piora real.

**Teste "justo" v2×v3 (mesma val, a do v3):** v2 deu AUC 0,976 / recall 96,2% — MAS isso é **vazamento**: o v2 tirou nota MAIOR na val do v3 (0,976) do que na val própria limpa dele (0,960), assinatura de memorização (treinou nesses recortes). Conclusão: não dá pra cravar v2×v3 por número sem retreinar os dois com split fixo compartilhado. Argumento que dispensa número: **v3 treinou em tudo do v2 + 28 casos difíceis**, então é ≥ v2 e mais robusto. **Decisão: adotar v3.**

**v3-final (produção) — montado 26/10/2026:** melhorei `fechar_crops.py` com `--out <dir>` (grava split em outro diretório, não sobrescreve) e `--sem-val` (modo produção: 100% dos dados no treino + val descartável duplicado do treino só pro YOLO salvar best.pt). Rodei `fechar_crops.py --crops data/crops_v2 --out data/crops_v3_final --sem-val` → **train 262d/2620s** (todos), val 52d/524s (duplicado). `data/crops_v2` (split limpo do v3) ficou intacto. **v3-final TREINADO (26/10/2026):** `runs/classify/agave_saude_v3_final/weights/best.pt` é o **modelo de produção** (60 épocas, val contaminado ~98% ignorado; métrica real = v3 limpo: AUC 0,940 / prec 88,6%). **Pipeline verificado de ponta a ponta** (`src/pipeline.py --det best_v5.pt --cls runs/classify/agave_saude_v3_final/weights/best.pt --device cpu`): numa foto deu 152 agaves / 7 doentes, JSON ok. Detector entregue = `best_v5.pt` (raiz) = `entrega_detector/modelo/best.pt` (idênticos). Pesos do pipeline são args de runtime (`--det`/`--cls`), não fixos no código; corrigi o docstring do pipeline.py (caminhos reais + `--device cpu`).

**Melhoria em aberto (alinhada ao produto de triagem):** o `pipeline.py` hoje devolve `saude`=top1 e `conf_saude`=top1conf. Pra ranquear suspeitas pro agrônomo o ideal é expor `prob_doente` em TODA planta (não só a vencedora). Recuperável como `conf_saude if saude=='doente' else 1-conf_saude`, mas melhor expor direto. NÃO feito ainda — decisão de schema do backend. Scripts de apoio (perdidos com scratchpad, recriar se precisar): `score_para_agronomo.py`, `montagem_lote.py`, `montagem_doente.py`, `incorporar_revisao.py`.

**(Histórico) Alternativa de garimpo:** com 234 doentes (alvo README 300), outra rodada com o v2 (AUC 0,960) exigiria fotos novas, que não há. Tweaks de modelo (yolov8m-cls etc.) são secundários. **Atenção ao treinar de novo:** rodar na CPU leva ~30min de compute, mas se a máquina dormir o processo em background pode ser morto por falta de memória — melhor o usuário rodar `train_cls.py` num terminal próprio, ou setar `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`. Mudança não commitada em `src/make_crops.py`: flag `--cls`/`--cls-device` (ranqueia pela prob do classificador em vez da cor). Scripts de apoio ainda não commitados: `tools/avaliar_cls.py`, `tools/aplicar_triagem.py`, `tools/conf_doente.py`.
