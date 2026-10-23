---
name: feedback-conducao-projeto
description: Como o usuário quer que eu trabalhe no agave-yolo — conduzir e melhorar scripts livremente
metadata:
  type: feedback
---

No [[project-agave-yolo-overview]], o usuário quer que eu **conduza** (proponha o próximo passo e execute, não só liste opções) e que **melhore os scripts livremente** quando achar um jeito melhor — autorizado a editar `src/` e `tools/` para melhorar o resultado.

**Why:** ele definiu a meta ("melhor classificador possível") e delegou o "como"; valoriza avançar com decisões técnicas, não ser consultado em cada micro-escolha.

**How to apply:** tomar decisões de ML sensatas por padrão, executar as partes automatizáveis, e deixar para ele só o que exige julgamento humano (ex.: triagem visual das doentes no Explorer) ou decisões de produto/cliente. Manter alterações de script retrocompatíveis (flags novas opcionais, não mudar o comportamento padrão).
