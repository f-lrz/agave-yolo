# Teste manual do detector

Para olhar o resultado do modelo em fotos novas, sem mexer em nada do pipeline
de treino.

## Como usar

1. Jogue as fotos em **`entrada/`**
2. Rode:

```bash
python teste_manual/testar.py
```

3. Olhe em **`saida/`**

Os arquivos de `entrada/` **nao sao modificados** — o script le os bytes e nunca
abre o original para escrita. Verificado por hash antes e depois.

## O que sai

| Caminho | O que e |
|---|---|
| `saida/imagens/<nome>_caixas.jpg` | a foto com as caixas desenhadas |
| `saida/labels/<nome>.txt` | label YOLO normalizado (`classe cx cy largura altura`) |
| `saida/json/<nome>.json` | saida completa: pixels + **confianca** de cada caixa |
| `saida/resumo.csv` | uma linha por foto: contagem, dimensao, tempo, avisos |

O `.txt` serve para abrir num anotador (Roboflow, CVAT, LabelImg). O `.json`
existe porque o formato YOLO **nao guarda a confianca** — se voce quiser filtrar
ou ordenar por confianca depois, e de la que vem.

O `resumo.csv` usa `;` como separador e BOM, para abrir direto no Excel em
portugues sem bagunçar as colunas.

## Opcoes

```bash
python teste_manual/testar.py --conf 0.20          # mais caixas, para pre-anotar
python teste_manual/testar.py --cor magenta        # foto saturada: preto some
python teste_manual/testar.py --device 0           # usar GPU (se houver)
python teste_manual/testar.py --refazer            # reprocessar o que ja tem saida
python teste_manual/testar.py --classe 0           # outro indice no .txt
```

Fotos que ja tem saida sao **puladas** por padrao. Assim voce joga uma foto nova
na pasta, roda de novo, e so a nova e processada — util porque cada foto leva
~11s.

## Sobre o `--conf`

O padrao e **0,30**, que e o calibrado para contagem: o erro na contagem fica
abaixo de 5%. Com `0,20` voce acha mais plantas (recall 0,870 contra 0,806) mas
a contagem infla ~28%.

| conf | recall | precisao | Erro na contagem |
|---|---|---|---|
| 0,20 | 0,870 | 0,681 | +27,8% |
| **0,30** | 0,806 | 0,768 | **+4,9%** |
| 0,40 | 0,717 | 0,822 | −12,8% |

## Duas coisas para olhar no resultado

**A coluna `avisos` do CSV.** Se vier preenchida, leia. O caso principal e foto
redimensionada: a deteccao cai pela metade sem dar erro nenhum.

**O modelo nao diz nada sobre doenca.** Todas as caixas sao a mesma classe
(`agave`). O `--classe 1` escreve `1` no `.txt` apenas para casar com a
convencao do projeto (`0 = doente`, `1 = sadia`), nao porque o modelo tenha
decidido que a planta esta sadia.

## Observacao

O script usa o mesmo codigo da entrega (`entrega_detector/agave_detector`), de
proposito: o que voce ve aqui e exatamente o que o backend vai produzir.
