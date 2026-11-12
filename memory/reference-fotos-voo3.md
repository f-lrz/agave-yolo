---
name: reference-fotos-voo3
description: 45 fotos do voo3 (10/11/2026) — split treino/teste/descartadas e onde está cada parte
metadata:
  type: reference
---

Origem: `C:\Users\Teste\Downloads\Fotos_10-11-2026_IDENTIFY2.0` (DJI_0079..0129, 45 fotos). Copiadas (originais intactos) para `data/novas_voo3/`, split **definido pelo usuário** em 11/11/2026:

- `descartadas_angulo/` (4): 0111, 0116, 0118, 0119 — ângulo diferente (oblíquo), fora de tudo.
- `teste/` (14): 0081, 85, 89, 92, 95, 102, 103, 106, 114, 117, 120, 125, 126, 127 — **holdout por foto: NUNCA recortar para treino.**
- `treino/` (27): as demais, usadas para garimpar doentes como no [[reference-fotos-novas-voo2]].

Nomes DJI_0079+ não colidem com voo1 (DJI_09xx, `imagem (N)`) nem voo2 (DJI_0041..0078). Uso no [[project-stage2-classifier]].
