# Entrega para o identify2.0-backend — v2.0.0 (deteccao + triagem de saude)

Esta pasta espelha a estrutura do `identify2.0-backend`: cada arquivo aqui vai
para o **mesmo caminho** la. Nada de dependencia nova: o `requirements.txt` do
backend continua igual.

## O que muda no backend

A API passa a fazer a etapa 2: alem de detectar as agaves, cada planta recebe
`prob_doente` (0 a 1), e a resposta traz a lista de plantas **ordenada da mais
suspeita para a menos**, que e a fila de revisao do agronomo.

**Compativel com o front atual.** Os campos que ja existiam continuam iguais
(`image_data`, `total_agaves`, `avisos`...). Os campos de saude sao novos, e o
front pode adota-los quando quiser. A unica mudanca visivel sem mexer no front e
a imagem: sadias em **ciano fino**, suspeitas em **vermelho grosso** com o
`prob_doente` escrito.

## Arquivos

| Desta pasta | Vai para o backend | |
|---|---|---|
| `app/services/agave_detector/saude.py` | mesmo caminho | **novo**: classificador de saude |
| `app/services/agave_detector/__init__.py` | mesmo caminho | substitui (versao 2.0.0, exporta o novo) |
| `app/services/agave_detector/detector.py` | mesmo caminho | substitui (+ aviso de foto tirada baixo demais) |
| `app/services/agave_detector/tiling.py` | mesmo caminho | igual ao atual (vai junto por completude) |
| `app/services/vision_service.py` | mesmo caminho | substitui: carrega os 2 modelos e devolve a saude |
| `app/api/routes.py` | mesmo caminho | substitui: campos novos na resposta |
| `app/main.py` | mesmo caminho | substitui: carrega os 2 modelos no startup; versao 2.0.0 |
| `models/classificador_saude.pt` | mesmo caminho | **novo**: pesos do classificador (10 MB) |
| `models/MODELO_SAUDE.md` | mesmo caminho | **novo**: ficha do classificador |
| `models/MODELO.md` | mesmo caminho | substitui (aponta para a ficha de saude; nova limitacao) |
| `API.md` | raiz | substitui: contrato com os campos de saude |
| `verificar_entrega.py` | raiz | temporario: confere a instalacao, apague depois |

**Nao muda:** `models/best.pt` (o detector e o mesmo), `requirements.txt`,
`app/core/`, `static/index.html` (continua funcionando, porque so le
`image_data`).

## Instalar (PowerShell)

Da raiz do `agave-yolo`:

```powershell
$B = "C:\Users\Teste\Documents\GitHub\identify2.0-backend"
Copy-Item entrega_backend\app\* "$B\app" -Recurse -Force
Copy-Item entrega_backend\models\* "$B\models" -Force
Copy-Item entrega_backend\API.md, entrega_backend\verificar_entrega.py "$B" -Force
```

Testado numa copia limpa do backend: os arquivos caem no lugar certo, sem criar
pasta duplicada.

## Conferir

Da raiz do backend, com o venv dele ativo e uma foto ORIGINAL do drone:

```powershell
cd C:\Users\Teste\Documents\GitHub\identify2.0-backend
python verificar_entrega.py C:\caminho\para\DJI_0126.JPG
```

O script confere o sha256 dos dois pesos, carrega os modelos como a API faz no
startup, analisa a foto e salva `verificacao_saida.jpg`. Saida esperada (CPU):

```
== pesos
  ok     models/best.pt
  ok     models/classificador_saude.pt
== carregando os modelos (como no startup da API)
  ok em 4.6s  (saude ativa: True, device: cpu, limiar: 0.5)
== analisando DJI_0126.JPG
  12.6s | 231 agaves | 1 suspeitas de doenca | taxa 0.0043
```

Depois, `uvicorn app.main:app` e um POST em `/api/analyze`, como antes.

## Variaveis de ambiente novas

Todas opcionais; os padroes ja sao os recomendados.

| Variavel | Padrao | Para que |
|---|---|---|
| `AGAVE_PESOS_SAUDE` | `models/classificador_saude.pt` | caminho dos pesos de saude |
| `AGAVE_SAUDE` | `1` | `0` desliga a etapa 2 (volta a ser so deteccao; campos de saude vem `null`) |
| `AGAVE_LIMIAR_DOENTE` | `0.5` | corte para `saude`/`doentes`. `0.3` acha mais doentes, com mais alarme falso |
| `AGAVE_COR` | `ciano` | cor das caixas sadias (**antes era `vermelho`**; o vermelho ficou para as doentes) |
| `AGAVE_COR_DOENTE` | `vermelho` | cor das caixas suspeitas |
| `AGAVE_MOSTRAR_SADIAS` | `1` | `0` desenha so as suspeitas |

`AGAVE_MOCK=1` continua funcionando, agora com uma planta sadia e uma doente
simuladas, no mesmo formato da resposta real.

## Testado antes da entrega

- Pacote nas 14 fotos de teste do voo de 10/11/2026 (nenhum modelo viu essas
  fotos): 8 suspeitas em 1.567 agaves; o classificador soma ~1 s por foto.
- API de ponta a ponta (`TestClient`) numa copia do backend com esta entrega
  aplicada: modo normal, `AGAVE_SAUDE=0`, `AGAVE_MOCK=1`, arquivo que nao e
  imagem (400) e JPEG corrompido (400).
- `verificar_entrega.py` rodando dentro da copia do backend.

## Antes de escrever texto para o usuario final

Leia `models/MODELO_SAUDE.md`. Em resumo: e **triagem**, nao diagnostico. Use
"suspeita de doenca — conferir em campo". Planta seca **nao** conta como doente.
Recall e precisao de ~81% na validacao, medidos numa amostra pequena e num sitio
so.
