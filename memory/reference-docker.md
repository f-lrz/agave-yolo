---
name: reference-docker
description: Conteinerização do Identify 2.0 (front + back) em Documents\Identify2.0_docker — arquitetura, decisões e estado
metadata:
  type: reference
---

`C:\Users\Teste\Documents\Identify2.0_docker` (16/11/2026). Decisão minha, a pedido do usuário: **2 containers separados + docker compose**, uma pasta com `backend/` e `frontend/`.

- `backend/`: `git archive` do `identify2.0-backend` (branch `feature/triagem-saude`) + `Dockerfile` (python:3.12-slim, torch de CPU via `--extra-index-url`, libgl1, usuário sem root, 1 worker) + `.dockerignore`. Imagem `identify2-backend:2.0.0`, 2,6 GB.
- `frontend/`: `git archive` do `identify-app` + `Dockerfile` multi-stage (node:22-alpine → nginx:1.27-alpine) + `nginx.conf` (proxy `/api/` e `/health` para `backend:8000`, `client_max_body_size 50m`, `proxy_read_timeout 300s`). Imagem 92 MB. Build com `VITE_API_URL=""` → o front chama a API no mesmo endereço; para isso mudei o front para `??` (commit `71b0c76`).
- Portas: `${PORTA:-8080}` (sistema, aberta na rede) e `127.0.0.1:${PORTA_API:-8000}` (só /docs, local).
- Testado: build, `compose up`, tudo via 8080 (fotos originais de 12,8 MB, 244/2 na DJI_0125), acesso pela rede local. ~18 s/foto no container contra ~13 s nativo.
- O código na pasta é CÓPIA dos repositórios. O `LEIA-ME.md` tem os comandos de atualização (`git archive -o` + `tar`; no PowerShell 5.1 o pipe corrompe binário).

Relacionado: [[project-stage2-classifier]], [[reference-demo-deteccao]].
