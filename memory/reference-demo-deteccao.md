---
name: reference-demo-deteccao
description: Pasta de demonstração da versão só-detecção do Identify 2.0 (back + front + pesos), montada em 15/11/2026
metadata:
  type: reference
---

`C:\Users\Teste\Documents\Identify2.0_demo_deteccao` — snapshot para APRESENTAR o projeto como estava quando só detectava agaves (o usuário chamou isso de "estágio 0"; no repositório, "estágio 0" é outra coisa: as propostas por CV sem modelo).

- `backend/`: cópia do `identify2.0-backend` no commit `ace728a` (06/10/2026, API 1.1.0), sem alterações; `models/best.pt` = `best_v5.pt` (sha256 a18006d5…); venv próprio já instalado (torch CPU, ~1,8 GB no total).
- `frontend/index.html`: a página `static/index.html` do backend, com `API_URL` apontando para `http://localhost:8000/api/analyze`. NÃO existe repositório de frontend do Identify nesta máquina; o usuário escolheu usar essa página como o front.
- `iniciar_backend.bat` (porta 8000), `iniciar_frontend.bat` (porta 5500), `LEIA-ME.md`, `fotos_exemplo/` (DJI_0030 → 110 agaves; exemplo_imagem31 → 115; 12-14 s cada, testado).

Relacionado: [[project-agave-yolo-overview]], [[project-stage2-classifier]].
