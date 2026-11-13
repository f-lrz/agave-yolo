# Ficha do modelo — classificador de saude

| | |
|---|---|
| Arquivo | `classificador_saude.pt` |
| sha256 | `da9834ce1f434ebf6de0d5aef0187f7de11d1b1de0fecabbda95afe9960964c4` |
| Arquitetura | YOLOv8s-cls (classificacao), entrada 224x224 |
| Classes | **2** — `doente` (indice 0), `sadia` (indice 1) |
| Data do treino | 11/11/2026 |
| Versao interna | v5-final (`agave_saude_v5_final` no repositorio `agave-yolo`) |

## Para que serve

**Triagem.** Para cada agave que o detector achou, devolve `prob_doente` (0 a
1): a chance de a planta estar doente, segundo o que o modelo aprendeu. A API
ordena as plantas por esse numero, e a lista vira a **fila de revisao do
agronomo**: comecar pelas mais suspeitas e conferir em campo.

## Para que NAO serve

- **Nao e diagnostico.** Nenhum laboratorio nem agronomo confirmou os rotulos
  de treino: eles vem de julgamento visual nas fotos aereas, recorte a recorte,
  pela equipe do projeto. O modelo aprendeu "parece doente vista de cima", e
  nao "esta doente".
- **Nao identifica QUAL doenca.** Sao duas classes so: doente e sadia.
- **Planta seca nao conta como doente.** Planta morta ou ressecada (tufo
  cinza/palha) foi ensinada como `sadia`, de proposito: ver "Convencao das
  secas" abaixo. Se o produto precisar contar plantas mortas, este modelo nao
  serve para isso.

Texto honesto para a tela: *"N plantas com suspeita de doenca — conferir em
campo"*. Evite "N plantas doentes" e "diagnostico".

## Como ele ve a planta

Cada caixa do detector vira um recorte com **15% de margem** de cada lado,
reduzido para 224x224 (`saude.py::recortar`). E exatamente o recorte usado no
treino. Mudar a margem ou o tamanho muda o que o modelo ve e degrada o
resultado sem dar erro.

## Dados de treino

| | |
|---|---|
| Fotos de origem | 107 fotos de drone de 3 voos (42 + 38 + 27), 5472x3078 |
| Doentes | **248** recortes |
| Sadias | 2.480 recortes (10 por doente), subamostrados de 23.791 |
| Escolha das sadias | metade **negativos dificeis** (as sadias que mais pareciam doentes no ranking da rodada anterior) e metade sorteada |
| Aumento de dados | rotacao livre (`degrees=180`), espelhamento nos dois eixos, variacao de cor HSV, `erasing=0.2` |
| Treino | 60 epocas, CPU |

A doenca e rara: cerca de **1 planta doente a cada 97**. Por isso as sadias
foram subamostradas: com todas, o modelo aprenderia a responder "sadia" sempre.

**Convencao das secas.** Na primeira versao, cerca de 33 plantas secas tinham
sido rotuladas como doentes, e o modelo passou a chamar toda planta seca de
doente. Elas foram tiradas do treino e guardadas como teste fixo de regressao.
O pool de sadias ja contem outras secas, entao o modelo aprende "seca nao e
doente" sem decorar essas 33.

## Desempenho medido

### Validacao (recortes de fotos que o modelo nao viu)

Medido no modelo `v5`, mesma receita do v5-final, mas com 53 doentes e 480
sadias separados **por foto de origem** para validacao. O v5-final treinou com
tudo, entao nao tem validacao propria; estes sao os numeros que o representam.

| corte (`prob_doente` >=) | acha | perde | alarme falso | recall | precisao |
|---|---|---|---|---|---|
| 0,10 | 47 | 6 | 19 | 0,887 | 0,712 |
| 0,30 | 45 | 8 | 13 | 0,849 | 0,776 |
| **0,50 (padrao)** | **43** | **10** | **10** | **0,811** | **0,811** |
| 0,70 | 39 | 14 | 9 | 0,736 | 0,812 |
| 0,90 | 33 | 20 | 7 | 0,623 | 0,825 |

**AUC 0,981.** Lendo como fila de revisao: os 10% mais suspeitos contem 43 das
53 doentes, e os 20% mais suspeitos contem 51 das 53.

Dois cuidados ao ler a tabela:

- **A precisao em campo e outra.** Na validacao ha 1 doente para cada 9
  sadias; no campo, 1 para cada ~97. Com muito mais sadias, os alarmes falsos
  pesam mais. Em compensacao, metade das sadias da validacao foi escolhida
  justamente entre as mais confusas, entao a taxa de alarme falso da tabela e
  pessimista para uma planta comum.
- **53 doentes e pouco.** Cada planta a mais ou a menos muda o recall em ~2
  pontos.

### Secas (teste de regressao fixo, 33 plantas que nenhum modelo viu)

| Versao | Secas chamadas de doente | prob_doente medio |
|---|---|---|
| v3-final (antes da correcao) | 29/33 | 0,898 |
| v4-final | 7/33 | 0,200 |
| **v5-final** | **3/33** | **0,115** |

### Fotos inteiras ineditas (14 fotos do voo de 10/11/2026)

Nenhum modelo viu essas fotos. O v5-final marcou **9 doentes**, contra 16 do
v4-final. As 9 do v5 estao todas entre as 16 do v4, e as 7 que so o v4 marcou
foram conferidas a olho: na maioria, alarmes falsos. **Quantas doentes as duas
versoes deixaram passar nessas fotos nao foi medido.**

## Escolhendo o corte

O `prob_doente` sai sempre. O corte (`AGAVE_LIMIAR_DOENTE`, padrao 0,5) so
decide o rotulo `saude` e a contagem `doentes`.

- **0,5 (padrao):** equilibrio, com ~81% de recall e ~81% de precisao na
  validacao.
- **0,3:** acha mais doentes (85%), com um pouco mais de alarme falso. Bom
  quando o agronomo prefere conferir mais plantas a perder uma.
- **0,7 ou mais:** so as mais certas. Menos alarme, mais doentes perdidas.

Para triagem, o mais util e a **ordem**: mostrar a lista do mais suspeito para
o menos, e deixar o agronomo descer ate onde quiser.

## Desempenho e dimensionamento

Cerca de **1 s por foto em CPU** (lote de 32 recortes; ~250 plantas por foto),
alem dos ~11 s do detector. Arquivo de ~10 MB e poucos MB de RAM a mais.

## Limitacoes — leia antes de prometer algo ao cliente

1. **Rotulos visuais, nao confirmados.** Ver "Para que NAO serve".
2. **Um sitio, um drone, uma epoca.** Treinado em uma plantacao, com o
   terreno, a luz e a fase da cultura desses voos. Em outro lugar ou outra
   estacao, espere desempenho pior sem retreinar.
3. **Depende do detector.** Planta que o detector nao acha nao e classificada.
   Foto tirada baixo demais (plantas acima de ~400px) fragmenta as caixas, e o
   recorte deixa de mostrar a planta inteira. O detector avisa nesse caso
   (campo `avisos`).
4. **Doenca em fase inicial.** O modelo so ve o que aparece de cima. Planta
   que adoeceu sem mudar de cor ou forma ainda nao e detectavel.

## Reproducao

Repositorio `agave-yolo`: `src/make_crops.py` (recortes) -> triagem manual ->
`tools/fechar_crops.py --sem-val` (dataset de producao) -> `src/train_cls.py`.
Avaliacao: `tools/avaliar_cls.py`. Historico completo das versoes em
`memory/project-stage2-classifier.md`.
