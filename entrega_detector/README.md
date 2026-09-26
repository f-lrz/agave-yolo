# Detector de agave — integracao no backend

Entrega parcial: **deteccao de agave em fotos de drone**. Uma caixa por planta,
mais a contagem total.

Este pacote foi enxugado para o `identify2.0-backend`, que ja e FastAPI. Nao ha
servico separado nem Dockerfile aqui: o que entra e a biblioteca e o modelo.

> **Este modelo nao diagnostica doenca.** Ele tem uma classe (`agave`) e nao
> distingue planta sadia de doente. O classificador de saude e uma etapa
> separada, ainda em desenvolvimento. Nenhuma tela deve rotular esta saida como
> "doente" ou "sadia".

## Instalacao

Copie `agave_detector/` e `modelo/` para dentro do backend, e acrescente as tres
linhas de [`requirements-adicional.txt`](requirements-adicional.txt) ao
`requirements.txt`. O que voce ja tem (fastapi, uvicorn, python-multipart,
opencv-python-headless, numpy) atende sem conflito de versao.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**O `--extra-index-url` nao e opcional.** Sem ele o pip instala o torch com
CUDA: **~4,5 GB** de bibliotecas de GPU que nunca serao usadas num servidor sem
placa. Com o indice de CPU, sao algumas centenas de MB.

## A armadilha que voce precisa conhecer antes de tudo

Isto **nao funciona**:

```python
from ultralytics import YOLO
YOLO('modelo/best.pt')('foto.jpg')      # acha quase nada
```

Uma foto de drone tem 5472x3078. O YOLO redimensiona a entrada para 1024px, o
que faz um agave de ~130px virar ~24px e sumir. A imagem **tem** que ser
recortada em tiles, cada tile inferido separadamente, e os resultados
remontados com NMS global.

O pacote `agave_detector` faz exatamente isso. **Use ele, nao o YOLO direto.**

## Uso

```python
from agave_detector import DetectorAgave

detector = DetectorAgave("modelo/best.pt")     # UMA vez, no startup do app

r = detector.detectar(bytes_do_upload)         # aceita path, bytes ou ndarray
r["total"]        # 678
r["caixas"][0]    # {"x1": 4510, "y1": 266, "x2": 4689, "y2": 443, "conf": 0.8721}
r["avisos"]       # []  -> ver secao "avisos" abaixo
```

Para a rota de download da imagem marcada:

```python
from agave_detector import desenhar_caixas

img = desenhar_caixas(bytes_originais, caixas_guardadas)   # nao carrega o modelo
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
```

[`exemplo_integracao.py`](exemplo_integracao.py) tem o padrao completo: singleton
do modelo, upload de varias fotos e a rota de download.

## As tres coisas que erram na integracao

**1. Carregar o modelo por requisicao.** Instancie `DetectorAgave` uma vez, no
`lifespan` do app, e reutilize. Carregar a cada chamada multiplica a latencia.

**2. Rodar a deteccao de novo no download.** Guarde as caixas que o `/detectar`
devolveu e use `desenhar_caixas()`. Sao milissegundos, contra os ~11s da
inferencia. E `desenhar_caixas()` nem precisa do modelo carregado.

**3. Processar varias fotos em paralelo.** Processe em **sequencia**. Cada foto
de 5472x3078 ocupa ~50 MB decodificada, e a inferencia ja usa todos os nucleos:
paralelizar multiplica a RAM sem ganhar tempo. A classe tem um lock interno que
serializa a inferencia, porque o FastAPI executa rota `def` num threadpool e o
modelo do Ultralytics nao e garantido thread-safe — mas conte com isso como
rede de seguranca, nao como estrategia.

## `avisos` — nao e decorativo

Se o campo vier preenchido, **mostre na interface ou registre no log**. O caso
principal e imagem redimensionada no caminho (front que comprime o upload,
thumbnail), e o efeito e grande sem que nenhum erro apareca:

| Imagem enviada | Deteccoes |
|---|---|
| 5472x3078 (original do drone) | **678** |
| 1600x900 (redimensionada) | **321** |

Metade das plantas desaparece. O detector avisa quando a imagem chega com menos
de 3000px de lado maior.

## Parametros

`tile`, `overlap` e `imgsz` fazem parte do **contrato do modelo** — estao fixos
em `agave_detector/detector.py` e mudar qualquer um degrada o resultado. Foram
medidos, nao escolhidos.

O unico parametro de uso e o **`conf`**:

| conf | Quando usar | recall | precisao | Erro na contagem |
|---|---|---|---|---|
| 0,20 | gerar rascunho de rotulo | 0,870 | 0,681 | +27,8% |
| **0,30** | **padrao — exibir ao usuario** | 0,806 | 0,768 | **+4,9%** |
| 0,40 | quando falso positivo incomoda mais que planta perdida | 0,717 | 0,822 | −12,8% |

O padrao de 0,30 e onde os falsos positivos quase compensam as plantas
perdidas, e a contagem erra menos de 5%. **Nao use 0,20 para exibir numero** —
inflaria a contagem em quase 30%.

## Desempenho e dimensionamento

Medido em CPU de 12 nucleos, imagem de 5472x3078:

| | |
|---|---|
| Carregar o modelo | < 1 s (uma vez) |
| **Inferencia por imagem** | **~11 s** |
| Desenhar + codificar JPEG | < 1 s |
| Tiles processados | 45 por imagem |
| RAM por processo | ~1,5 GB incluindo o modelo |
| JPEG marcado, resolucao original | ~6,6 MB |

Consequencias:

- **11 s nao cabe numa requisicao sincrona**, e com varias fotos e pior: 10
  fotos sao ~2 minutos, o que estoura o timeout de qualquer proxy. Trate como
  job: enfileira, devolve o id, o front consulta o progresso.
- Um worker por processo. A inferencia ja ocupa todos os nucleos; mais workers
  competem entre si e cada um carrega sua propria copia do modelo. Para mais
  vazao, escale em replicas.
- Com GPU, passe `device="0"` ao construtor. Cai para menos de 1 s por imagem.

## Limitacoes — leia antes de prometer algo ao cliente

Os numeros acima foram medidos em **2 imagens de validacao** (756 plantas). Sao
honestos, mas a amostra e pequena: nao trate o erro de contagem de 4,9% como
garantia em um voo novo.

O modelo foi treinado numa plantacao especifica, com um drone especifico. Em
outro sitio, outra cultura ou outra camera, espere desempenho pior sem
retreinar.

A ficha completa esta em [`modelo/MODELO.md`](modelo/MODELO.md). Vale a leitura
de quem for escrever o texto que o cliente vai ver.

## Estrutura

```
entrega_detector/
  README.md                    este arquivo
  requirements-adicional.txt   as 3 linhas a somar ao requirements do backend
  exemplo_integracao.py         padrao de service + rotas (detectar e download)
  agave_detector/
    __init__.py
    detector.py                DetectorAgave, desenhar_caixas, ler_imagem
    tiling.py                  recorte em tiles
  modelo/
    best.pt                    os pesos
    MODELO.md                  ficha do modelo
```
