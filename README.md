# Detecção de agaves doentes em imagens de drone

Pipeline de 2 estágios:

```
imagem de drone (4000x3000)
    |
    |-- ESTÁGIO 1: YOLOv8 detect (1 classe: agave)  -> onde estão as plantas
    |
    `-- ESTÁGIO 2: YOLOv8-cls (sadia | doente)      -> estado de cada planta
```

**Por que 2 estágios e não um detector de 2 classes:** as doentes serão minoria
(talvez 5-15% do plantio). Num detector único, a classe rara vira caixa perdida
— a planta doente simplesmente não é detectada, que é justamente o erro que você
não pode cometer. Separando, o estágio 1 acha *toda* planta (problema fácil,
muitos exemplos) e o estágio 2 só decide o estado. Bônus: dá para mexer no
limiar de doença sem retreinar a detecção, e o estágio 2 reaproveita 100% das
caixas do estágio 1.

---

## Instalação

Comece descobrindo o que já existe na máquina — o verificador diz o comando
exato do que falta:

```bash
python tools/checar_ambiente.py
```

**O estágio 0 não precisa de PyTorch nem de GPU.** Só isto:

```bash
pip install numpy opencv-python tqdm
```

Com isso você já roda `cv_proposals.py` e `eval_proposals.py` e começa a montar
o dataset hoje. PyTorch e Ultralytics só entram na hora de treinar:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install ultralytics
```

> **A pegadinha do Windows:** `pip install torch` sem o `--index-url` instala a
> build **só de CPU**, sem avisar. Ela importa, roda e treina cerca de 30x mais
> devagar, e a sua 1660 fica parada. O `checar_ambiente.py` detecta isso
> (`torch.version.cuda = None`) e mostra como reinstalar.

> **Atenção GTX 16xx:** a série 1660 é Turing **sem Tensor Cores**. Mixed
> precision (AMP) dá NaN no loss ou fica mais lenta que FP32. Todos os scripts
> daqui já usam `amp=False` por padrão. Não ligue.

---

## Estrutura

```
data/raw/images/               suas fotos de drone originais (.jpg)
data/raw/labels/               .txt YOLO, mesmo nome-base da imagem
data/raw/nao_rotuladas/        o resto do voo, a ser pré-anotado
data/tiles/                    dataset de tiles gerado
data/crops/                    recortes do estágio 2
src/cv_proposals.py            estagio 0: propostas sem treinar nada
src/eval_proposals.py          mede propostas contra os seus labels
src/                           o resto do pipeline
tools/checar_ambiente.py       verifica dependencias e GPU
tools/checar_dados.py          verifica as pastas antes de rodar
tools/juntar_classes.py        colapsa 2 classes em 1 para o estagio 1
tools/gerar_dados_falsos.py    imagens sintéticas para testar o fluxo
```

Formato do label (YOLO, normalizado): `classe cx cy largura altura`, um por
linha. Se você rotulou no Roboflow/CVAT, exporte como **YOLOv8**.

```bash
python tools/checar_dados.py
```

Roda essa verificação antes de qualquer coisa: ela pega nome de `.txt` que não
bate com a imagem, coordenadas em pixels absolutos em vez de normalizadas, e
imagem repetida entre as pastas.

### Se você rotulou com 2 classes (sadia / doente)

O detector do estágio 1 precisa de **uma classe só** — ele tem que achar todo
agave, doente ou não. Colapse antes de gerar os tiles:

```bash
python tools/juntar_classes.py --entrada data/raw/labels --saida data/raw/labels_1classe
```

E use `--labels data/raw/labels_1classe` no `make_tiles.py`. Os labels originais
de 2 classes ficam intactos — são eles que alimentam o estágio 2 depois. Funciona
com qualquer convenção de ids: tudo vira classe 0.

Registre qual id é qual num `data/raw/classes.txt`, um nome por linha, na ordem
dos ids (linha 1 = classe 0):

```
doente
sadia
```

É a convenção do LabelImg, e o `checar_dados.py` passa a mostrar os nomes junto
com a contagem. Em dois meses ninguém lembra se 0 era doente ou sadia.

**Ao pré-anotar, emita a classe majoritária.** As propostas saem com um id fixo;
se ele não bater com a sua convenção, você abre o CVAT e vê centenas de plantas
marcadas com a classe errada:

```bash
python src/cv_proposals.py --images ... --out ... --save-labels --classe 1
```

Com 0=doente e 1=sadia, use `--classe 1`: quase toda proposta já fica certa, e
na correção você só vira as poucas doentes. O `predict_tiled.py` tem a mesma
opção, para as pré-anotações do YOLO semente.

> **Por que isso não é opcional:** com labels de classe 1 e um `data.yaml` de uma
> classe só, o Ultralytics **não dá erro**. Ele imprime `ignoring corrupt
> image/label: Label class 1 exceeds dataset class count 1` e **descarta o tile
> inteiro**, junto com todos os agaves saudáveis que estavam nele. O treino
> segue, a mAP sai com cara de normal, e os únicos tiles perdidos são
> exatamente os que continham plantas doentes.

---

## ESTÁGIO 0 — arrancar sem treinar nada (visão computacional clássica)

Serve para gerar as primeiras centenas de caixas sem depender de nenhum modelo
treinado. Explora três propriedades do problema: agave é verde sobre solo
marrom, a roseta é um blob aproximadamente circular, e o plantio é espaçado.

```bash
python src/cv_proposals.py --images data/raw/amostra --out out/cv --debug
```

O `--debug` salva um painel com as 6 etapas intermediárias. **Olhe esse painel
antes de mexer em qualquer parâmetro** — ele diz em qual etapa você está
perdendo planta, e evita ajustar filtro de forma quando o problema está na
segmentação.

Depois de achar os parâmetros, meça contra suas imagens rotuladas:

```bash
python src/eval_proposals.py --pred out/cv/labels --gt data/raw/labels --images data/raw/images --out out/erros
```

Aqui é o melhor uso das suas 9 imagens: como conjunto de **medida**, não de
treino. `--out` desenha os erros — verde = acerto, vermelho = planta que faltou,
azul = caixa sobrando.

Quando estiver bom, rode em tudo:

```bash
python src/cv_proposals.py --images data/raw/nao_rotuladas --out data/preanot --plant-px 130 --save-labels
```

### O parâmetro que mais importa: `--thresh-scale`

O limiar de Otsu puro é **agressivo demais** para este problema. Ele assume duas
populações (solo e planta) e corta no vale entre elas — mas a planta amarelada
cai justamente no vale e desaparece. Como o objetivo é pré-anotar, e apagar
caixa é muito mais rápido que desenhar a que faltou, o padrão corta em 0.40 do
Otsu.

Medido no conjunto sintético (`tools/gerar_dados_falsos.py`, 4 imagens, 206
plantas), variando só esse parâmetro:

| `--thresh-scale` | recall | precisão | F1 |
|---|---|---|---|
| 0.25 | 0.908 | 0.760 | 0.827 |
| 0.30 | 0.922 | 0.809 | **0.862** |
| **0.40 (padrão)** | 0.893 | 0.807 | 0.848 |
| 0.55 | 0.879 | 0.812 | 0.844 |
| Otsu puro (~1.0) | 0.762 | 0.835 | 0.797 |

Abaixo de ~0.25 a máscara inunda e tudo colapsa (o script detecta isso e avisa).

### O viés que você precisa conhecer

O índice de vegetação **perde justamente as plantas doentes**. Medido no mesmo
conjunto: com todas as plantas vigorosas o recall é **0.951**; incluindo plantas
amareladas/amarronzadas ele cai para **0.762** com Otsu puro. Faz sentido — uma
agave morrendo é marrom, ou seja, da cor do solo.

Isso importa muito no seu caso, porque cria um **viés de seleção**: se você
rotular apenas o que o detector clássico propõe, o YOLO treinado em cima herda
o ponto cego exatamente na classe que o projeto existe para achar.

Duas defesas, use as duas:

1. Baixe o `--thresh-scale` (é o que a tabela acima mostra recuperando recall).
2. Na passada de correção, **procure ativamente por plantas amareladas que não
   receberam caixa**. Não confie no que a ferramenta propôs — o que ela erra não
   é aleatório, é enviesado contra o seu alvo.

---

## ESTÁGIO 1 — detector de agave

### Passo 0. Meça o tamanho dos agaves (faça isso primeiro)

```bash
python src/inspect_labels.py --images data/raw/images --labels data/raw/labels --imgsz 1024
```

Isso decide o `--tile` de todo o resto. Um agave de 130px numa imagem de 4000px
some se você jogar a imagem inteira num YOLO de 1024px (vira 33px). Recortando
em tiles de 1024, ele chega na rede com os 130px originais.

### Passo 1. Gerar o dataset de tiles

```bash
python src/make_tiles.py --images data/raw/images --labels data/raw/labels --out data/tiles --tile 1024 --overlap 0.2
```

Suas 9 imagens viram ~180 tiles. O split treino/val é feito **por imagem de
origem**, nunca por tile — tiles vizinhos se sobrepõem, e embaralhar tiles vaza
dados da validação para o treino (a mAP fica linda e o modelo não funciona em
campo).

Com poucas imagens, escolha você mesmo quais vão para validação:

```bash
python src/make_tiles.py --images data/raw/images --labels data/raw/labels --out data/tiles --val-images voo_07 voo_08
```

### Passo 2. Treinar o detector semente

```bash
python src/train_det.py --data data/tiles/data.yaml --preset gtx1660 --epochs 150 --name agave_v1
```

Com 9 imagens o resultado será médio (espere mAP50 de 0.5-0.7). Tudo bem: o
objetivo deste modelo não é produzir laudo, é **pré-anotar o resto para você**.

Se der `CUDA out of memory`: `--batch 2`, ou `--model yolov8n.pt`, ou `--imgsz 768`.

### Passo 3. Pré-anotar as imagens não rotuladas

```bash
python src/predict_tiled.py --weights runs/detect/agave_v1/weights/best.pt --images data/raw/nao_rotuladas --out data/preanot --tile 1024 --conf 0.20 --save-labels
```

Use `--conf` **baixo** (0.15-0.25) aqui. Apagar uma caixa errada leva 1 segundo;
desenhar uma que faltou leva 10. Melhor errar para mais.

Saem `.txt` nas coordenadas da imagem inteira. Importe imagem + txt no CVAT (ou
LabelImg, ou Roboflow) e **corrija** em vez de desenhar do zero.

### Passo 4. Repetir

Junte as corrigidas em `data/raw/`, volte ao passo 1. Progressão típica:

| rodada | imagens rotuladas | instâncias | esforço                     |
|--------|-------------------|-----------|------------------------------|
| 0      | 9                 | ~450      | manual, já feito             |
| 1      | +20               | ~1.400    | corrigir pré-anotação (~1h)  |
| 2      | +40               | ~3.400    | corrigir (~1h30)             |
| 3      | +60               | ~6.400    | só revisar                   |

Perto de **3.000-5.000 instâncias** o detector estabiliza (mAP50 > 0.90).

Da rodada 2 em diante, priorize imagens de condição *diferente* — outro horário,
outro talhão, dia nublado. 10 imagens variadas valem mais que 50 iguais; o que
mais quebra modelo de drone é mudança de iluminação.

---

## ESTÁGIO 2 — sadia vs. doente

### Passo 5. Recortar cada planta

```bash
python src/make_crops.py --images data/raw/images --labels data/raw/labels --out data/crops
```

Cada caixa vira um JPG 224x224 em `data/crops/_classificar/`. **Você não desenha
nenhuma caixa nova.** Rotular vira arrastar arquivo entre duas pastas no
Explorer — dá para fazer centenas por hora, e um agrônomo consegue revisar sem
aprender ferramenta de anotação.

Separe assim (val de imagens de origem DIFERENTES das do train):

```
data/crops/train/sadia/    data/crops/train/doente/
data/crops/val/sadia/      data/crops/val/doente/
```

### Passo 6. Treinar o classificador

```bash
python src/train_cls.py --data data/crops --model yolov8s-cls.pt --epochs 60 --batch 32
```

Cabe folgado nos 6GB (224x224 é barato). Meta: **pelo menos 300 recortes da
classe doente**. Abaixo disso o modelo decora em vez de aprender.

Olhe a **matriz de confusão** em `runs/classify/agave_saude/`, não só a acurácia.
Com 10% de doentes, um modelo que chuta "sadia" sempre acerta 90% e é inútil. O
que importa é o **recall da classe doente**: das plantas doentes, quantas você
achou.

---

## ESTÁGIO 3 — usar no backend

```bash
python src/pipeline.py --det runs/detect/agave_v1/weights/best.pt --cls runs/classify/agave_saude/weights/best.pt --images data/raw/novas --out out/laudo
```

Gera um `.jpg` anotado e um `.json` por imagem:

```json
{
  "total_agaves": 48,
  "doentes": 6,
  "sadias": 42,
  "taxa_doenca": 0.125,
  "plantas": [
    {"bbox": [3556.3, 694.5, 3719.8, 866.0],
     "conf_deteccao": 0.93, "saude": "doente", "conf_saude": 0.81}
  ]
}
```

No seu backend, carregue os pesos **uma vez** no startup (carregar por requisição
custa ~2s):

```python
from pipeline import AgavePipeline

pipe = AgavePipeline("pesos/det.pt", "pesos/cls.pt", device="0")   # no startup
laudo = pipe.run(img_bgr)                                          # por requisição
img_anotada = pipe.annotate(img_bgr, laudo)
```

---

## Testar o fluxo sem os seus dados

```bash
python tools/gerar_dados_falsos.py
```

Cria 9 imagens sintéticas 4000x3000 com 48 "agaves" cada, com labels. Serve para
validar que a instalação está certa antes de gastar tempo com dados reais.

---

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `Label class 1 exceeds dataset class count 1` | labels de 2 classes num dataset de 1 | rode `tools/juntar_classes.py` — não ignore, o tile inteiro é descartado em silêncio |
| cv_proposals acha 0 caixas | `plant_px` estimado errado; o filtro de área rejeita tudo | passe `--plant-px` medido por você; o script diz quantas foram rejeitadas por área |
| cv_proposals: "vegetação ocupa 90%" | `--thresh-scale` baixo demais, a máscara inundou | suba para 0.5–0.6 até a cobertura ficar entre 3% e 25% |
| cv_proposals pega muito mato | filtros de forma frouxos | suba `--min-solidity` (0.65) e `--min-area` |
| uma planta vira 2 caixas | `--peak-sep` baixo demais | suba para 0.55–0.7 |
| duas plantas viram 1 caixa | `--peak-sep` alto demais | baixe para 0.35 |
| `torch.version.cuda = None` | build de CPU do PyTorch instalada | reinstale com `--index-url .../cu121` |
| `CUDA out of memory` | batch/imgsz grande demais | `--batch 2`, `--model yolov8n.pt`, ou `--imgsz 768` |
| loss vira `nan` | AMP na GTX 16xx | confirme `amp=False` (já é o padrão) |
| treino trava no início no Windows | dataloader workers | `workers=2` ou `0` |
| mAP alta no treino, péssimo em campo | split por tile, não por imagem | use `--val-images` com imagens de outro voo |
| detecta a mesma planta 2x | costura entre tiles | aumente `--iou` no NMS global, ou reduza `--overlap` |
| não detecta nada | conf alto demais para um modelo fraco | teste `--conf 0.05` antes de achar que é bug |
| plantas da borda cortadas ao meio | `min_visibility` baixo | suba `--min-visibility 0.5` no make_tiles |
