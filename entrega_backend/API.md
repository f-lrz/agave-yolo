# API — IDENTIFY 2.0 backend

Contrato para quem vai consumir a API no frontend.

> ## Leia isto antes de desenhar a tela
>
> A API faz duas coisas:
>
> 1. **Detecta as agaves**: uma caixa por planta, e a contagem.
> 2. **Faz a triagem de saúde**: cada planta recebe `prob_doente` (0 a 1), a
>    chance de estar doente segundo o modelo.
>
> **A triagem não é diagnóstico.** Ela aponta quais plantas o agrônomo deve
> conferir em campo, começando pelas mais suspeitas. Os rótulos de treino vêm de
> julgamento visual nas fotos aéreas, sem confirmação de laboratório.
>
> Rótulo honesto para a tela: *"N plantas com suspeita de doença — conferir em
> campo"*. Evite "N plantas doentes", "diagnóstico" e "laudo".
>
> Planta **seca** (morta, ressecada) **não** conta como doente: o modelo foi
> ensinado assim de propósito.
>
> Detalhes e limitações: [`models/MODELO.md`](models/MODELO.md) (detector) e
> [`models/MODELO_SAUDE.md`](models/MODELO_SAUDE.md) (saúde).

---

## `GET /health`

```json
{ "status": "online", "message": "Backend do IDENTIFY 2.0 ativo." }
```

## `POST /api/analyze`

Uma imagem por requisição. `multipart/form-data`, campo **`file`**.

```js
const formData = new FormData();
formData.append("file", arquivo);
const resp = await fetch("/api/analyze", { method: "POST", body: formData });
const data = await resp.json();
```

### Resposta `200`

Exemplo (a lista `plantas` foi encurtada; na resposta real vêm todas):

```json
{
  "filename": "DJI_0125.JPG",
  "content_type": "image/jpeg",
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
  "total_agaves": 244,
  "largura": 5472,
  "altura": 3078,
  "conf": 0.3,
  "avisos": [],
  "saude_ativa": true,
  "limiar_doente": 0.5,
  "doentes": 2,
  "sadias": 242,
  "taxa_doenca": 0.0082,
  "plantas": [
    {"x1": 3441, "y1": 596, "x2": 3623, "y2": 763, "conf": 0.5549,
     "prob_doente": 0.9998, "saude": "doente"},
    {"x1": 1201, "y1": 2210, "x2": 1398, "y2": 2401, "conf": 0.7012,
     "prob_doente": 0.654, "saude": "doente"},
    {"x1": 4012, "y1": 88, "x2": 4230, "y2": 301, "conf": 0.8123,
     "prob_doente": 0.0015, "saude": "sadia"}
  ]
}
```

| Campo | Tipo | O que é |
|---|---|---|
| `filename` | string | o nome que veio no upload |
| `content_type` | string | o MIME que veio no upload |
| `image_data` | string | **data-URI pronto para `<img src>`**: a foto com as caixas desenhadas (sadias em ciano fino, suspeitas em vermelho grosso com o `prob_doente` escrito) |
| `total_agaves` | int | quantas plantas foram detectadas |
| `largura`, `altura` | int | dimensões da imagem enviada, em pixels |
| `conf` | float | limiar do detector (ver "Sobre o `conf`") |
| `avisos` | string[] | **se não estiver vazio, mostre ao usuário** (ver abaixo) |
| `saude_ativa` | bool | `false` se o servidor rodar só com detecção (`AGAVE_SAUDE=0`); aí os campos de saúde vêm `null` |
| `limiar_doente` | float | corte usado para `saude` e `doentes` (padrão 0,5) |
| `doentes` | int | plantas com `prob_doente >= limiar_doente`: **suspeitas**, não confirmadas |
| `sadias` | int | o resto |
| `taxa_doenca` | float | `doentes / total_agaves`, de 0 a 1 |
| `plantas` | object[] | **todas** as plantas, **ordenadas da mais suspeita para a menos** |

Cada item de `plantas`:

| Campo | Tipo | O que é |
|---|---|---|
| `x1`, `y1`, `x2`, `y2` | int | a caixa, em pixels da imagem enviada (origem no canto superior esquerdo) |
| `conf` | float | confiança do **detector** de que existe uma agave ali |
| `prob_doente` | float | 0 a 1: suspeita de doença. **É o número da triagem** |
| `saude` | string | `"doente"` ou `"sadia"`, só aplicando o corte `limiar_doente` ao `prob_doente` |

Não confunda `conf` com `prob_doente`. `conf` diz se existe planta ali;
`prob_doente` diz se a planta parece doente.

O `image_data` já serve tanto para exibir quanto para o botão de download:

```js
const link = document.createElement("a");
link.href = data.image_data;
link.download = `agaves_${data.filename}`;
link.click();
```

### Erros

| Código | Quando | `detail` |
|---|---|---|
| `400` | não é imagem, arquivo vazio, ou imagem corrompida | mensagem em português, exibível |
| `422` | o campo `file` não foi enviado | formato padrão do FastAPI |
| `500` | falha inesperada no processamento | mensagem genérica |

O corpo do erro é `{ "detail": "..." }`. As mensagens de `400` são escritas para
serem mostradas ao usuário direto.

---

## A tela de triagem

O jeito certo de usar a saúde é uma **fila de revisão**, não um número solto:

- `plantas` já vem ordenada. Mostre as primeiras (ex.: as 10 mais suspeitas,
  ou todas com `prob_doente >= 0.3`) com um recorte da foto em volta da caixa,
  para o agrônomo marcar "confirmei" ou "descartei".
- A planta com `prob_doente` 0,45 aparece como `"sadia"` pelo corte de 0,5,
  mas é justamente a que mais vale olhar. Se a tela tiver um controle de
  sensibilidade, filtre por `prob_doente` no front, sem nova requisição.
- Na maioria das fotos `doentes` é 0, 1 ou 2: a doença é rara (cerca de 1
  planta a cada 100). Mostrar "0 suspeitas" é um resultado normal, não erro.

Para desenhar as caixas no front em vez de usar o `image_data`, as
coordenadas estão em pixels da imagem **original**. Escale pela razão entre o
tamanho exibido e `largura`/`altura`.

## Três coisas que vão afetar a sua tela

### 1. Demora ~12 segundos por imagem

Não é exagero nem bug: a imagem de drone é recortada em 45 pedaços e cada um
passa pelo detector (~11 s); depois cada planta passa pelo classificador de
saúde (~1 s). Medido em CPU de 12 núcleos.

Consequências:

- **Precisa de estado de carregamento.** 11 segundos sem retorno visual parece
  travamento.
- **Com várias fotos, multiplica.** 10 fotos = ~2 minutos. Envie uma por
  requisição, em sequência, e mostre o progresso por foto (`3 de 10`) em vez de
  uma barra única. Não dispare as 10 em paralelo: o servidor processa uma por
  vez de qualquer forma (há uma trava interna), então o paralelismo só faria
  todas estourarem o timeout juntas.
- **Cuidado com o timeout** do `fetch`, do proxy e do gateway. 11s por imagem já
  passa do padrão de alguns.

### 2. A resposta pesa ~7,6 MB por imagem

O `image_data` é a imagem em resolução original (5472×3078) embutida em base64,
o que infla o tamanho em ~33%.

- Em localhost é instantâneo. Pela internet, não.
- Não guarde muitas dessas strings em memória no navegador: 10 fotos = ~76 MB
  de JavaScript heap.
- A lista `plantas` é leve em comparação: ~100 bytes por planta, cerca de
  25 KB numa foto com 250 plantas.
- Se isso virar problema, fale com o backend: a alternativa é a API salvar a
  imagem e devolver uma URL. Muda o contrato, então é conversa, não ajuste.

### 3. O campo `avisos` não é decorativo

Hoje existem dois avisos, e os dois importam.

#### Imagem redimensionada

O modelo foi treinado em fotos de 5472×3078, onde a planta tem ~130px. Se a
imagem chegar reduzida, as plantas ficam pequenas demais e a detecção despenca
— **sem gerar erro nenhum**:

| Imagem enviada | Plantas detectadas |
|---|---|
| 5472×3078 (original do drone) | **678** |
| 1600×900 (redimensionada) | **321** |

Metade desaparece silenciosamente. Então:

- **Não redimensione nem recomprima a imagem no frontend antes de enviar.**
  Mande o arquivo original do drone.
- Se `avisos` vier preenchido, mostre o texto. Ele já vem escrito em português
  e explica o que o usuário deve fazer.

#### Foto tirada baixo demais

O modelo aprendeu plantas de ~110 a ~320 px. Se o drone voou mais baixo e as
plantas aparecem com 400 px ou mais, o detector fragmenta umas e perde outras,
e a contagem e a triagem deixam de ser confiáveis. Aconteceu em 4 das 14 fotos
de teste do voo de 10/11/2026. Nesse caso vem um aviso começando por *"as plantas
aparecem com ~656px nesta foto"*: mostre-o junto do resultado.

---

## Sobre o `conf`

É o limiar de confiança. Está fixo em **0,30** no servidor, e esse valor foi
escolhido medindo:

| conf | Plantas achadas | Caixas erradas | Erro na contagem |
|---|---|---|---|
| 0,20 | 87% | muitas | +27,8% |
| **0,30** | 81% | moderadas | **+4,9%** |
| 0,40 | 72% | poucas | −12,8% |

Em 0,30 as caixas sobrando quase compensam as plantas perdidas, e a **contagem
exibida erra menos de 5%**. É por isso que ele é o padrão, e é por isso que o
número em `total_agaves` pode ser apresentado como estimativa de contagem.

O campo vem na resposta para registro; o frontend não escolhe o valor.

---

## Honestidade sobre a precisão

### Detecção (contagem)

Os números acima foram medidos em **2 imagens de validação** (756 plantas
contadas à mão). São honestos, mas é uma amostra pequena: não trate o erro de
4,9% como garantia num voo novo, e evite exibir a contagem sem a palavra
"aproximadamente".

O modelo foi treinado numa plantação específica, com um drone específico. Em
outro sítio ou outra câmera, o desempenho cai.

### Saúde (triagem)

Validação em recortes de fotos que o modelo não viu (53 plantas doentes):

| Corte | Doentes achadas | Das marcadas, quantas eram doentes |
|---|---|---|
| 0,3 | 85% | 78% |
| **0,5 (padrão)** | **81%** | **81%** |
| 0,7 | 74% | 81% |

Na prática: de cada 5 plantas doentes, o modelo aponta cerca de 4, e cerca de
1 em cada 5 suspeitas é alarme falso. Ordenando por `prob_doente`, os 20% mais
suspeitos contêm 51 das 53 doentes. Por isso a fila de revisão funciona melhor
que o corte fixo.

Amostra pequena, um sítio só, rótulos visuais. Os números completos estão em
[`models/MODELO_SAUDE.md`](models/MODELO_SAUDE.md).
