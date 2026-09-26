# Ficha do modelo

| | |
|---|---|
| Arquivo | `best.pt` |
| sha256 | `a18006d57162cb8d3e163751321fb3d322ca47ec406a419906e254857292e96e` |
| Arquitetura | YOLOv8s (deteccao) |
| Classes | **1** — `agave` |
| Data do treino | 26/09/2026 |
| Versao interna | v5 |

## Para que serve

Localizar plantas de agave em fotos aereas de drone. Uma caixa por planta.

## Para que NAO serve

**Nao diagnostica doenca.** O modelo tem uma classe so. Ele nao distingue planta
sadia de planta doente, e nao ha como extrair essa informacao da saida dele.
O classificador de saude e uma etapa separada, ainda em construcao.

Se alguma tela exibir "doente" com base neste modelo, esta errada.

## Dados de treino

| | |
|---|---|
| Imagens | 42 fotos de drone, 5472x3078 |
| Caixas | 14.568, todas rotuladas a mao |
| Tiles de treino | 1.625 (35.861 caixas com a duplicacao da sobreposicao) |
| Tiles de validacao | 90 (2 imagens separadas, nunca vistas no treino) |
| Origem | uma plantacao, voos em alturas diferentes |

Hiperparametros: `imgsz=1024`, `batch=8`, `50 epocas`, AMP ligado, augmentacao
para vista de cima (`degrees=180`, `flipud=0.5`, `scale=0.5`, variacao de HSV).

## Desempenho medido

Medido nas **2 imagens de validacao** (756 plantas), com o pipeline completo de
tiling + NMS — nao por tile.

| conf | recall | precisao | F1 | Caixas previstas | Erro na contagem |
|---|---|---|---|---|---|
| 0,20 | 0,870 | 0,681 | 0,764 | 966 | +27,8% |
| **0,30** | **0,806** | **0,768** | **0,786** | **793** | **+4,9%** |
| 0,40 | 0,717 | 0,822 | 0,766 | 659 | −12,8% |
| 0,50 | 0,594 | 0,884 | 0,710 | 508 | −32,8% |

Por tile (metrica do Ultralytics): `mAP50 = 0,867`, `mAP50-95 = 0,519`.

**`conf = 0,30` e o padrao** porque e onde os falsos positivos quase compensam
as plantas perdidas: a contagem erra menos de 5%. Use `0,20` apenas para gerar
rascunho de rotulo, nunca para exibir numero.

## Limitacoes — leia antes de prometer algo ao cliente

1. **A validacao sao 2 imagens.** Os numeros acima tem margem de erro grande.
   Nao trate o +4,9% de erro de contagem como garantia em um voo novo.
2. **Um sitio, um drone.** Treinado numa plantacao especifica, com terreno de
   capim seco e solo avermelhado. Em outra cultura, outro solo ou outra camera,
   espere desempenho pior sem retreinar.
3. **Exige a resolucao original.** O modelo aprendeu plantas de ~110 a ~320px
   dentro de tiles de 1024. Uma imagem redimensionada no caminho (upload
   comprimido pelo front, thumbnail) quebra isso e a deteccao cai muito. O
   detector emite aviso quando a imagem chega com menos de 3000px.
4. **Rotulos feitos por nao-especialistas.** O convencionado e que a caixa cobre
   o leque de folhas, mas o tamanho medio variou entre rodadas (130px nas
   primeiras, 110px nas ultimas). Uma das 42 imagens (`imagem (22)`) foi
   revisada apenas superficialmente.
5. **Nao foi testado em imagem diagonal.** Uma foto do voo foi descartada por
   ter sido tirada em angulo; o modelo nunca viu esse caso.

## Reproducao

O treino sai do repositorio `agave-yolo`, com `src/make_tiles.py` +
`src/train_det.py`. Parametros do dataset: `--tile 1024 --overlap 0.4
--min-visibility 0.6 --empty-ratio 0.25`.
