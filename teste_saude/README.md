# Teste manual do pipeline de saude

Igual ao `teste_manual/` (que testa so o detector), mas aqui roda o **pipeline
completo**: detecta as agaves **e** classifica cada uma em sadia/doente. A foto
de saida mostra **so as caixas vermelhas de doente** — as sadias nao sao
desenhadas, para a doenca saltar aos olhos.

## Como usar

1. Jogue as fotos em **`entrada/`**
2. Rode:

```bash
python teste_saude/testar.py
```

3. Olhe em **`saida/`**

Os arquivos de `entrada/` **nao sao modificados** — o script le os bytes e
decodifica em memoria, nunca abre o original para escrita.

## O que sai

| Caminho | O que e |
|---|---|
| `saida/imagens/<nome>_doentes.jpg` | a foto com SO as caixas de doente (vermelho), cada uma com o `prob_doente` escrito |
| `saida/json/<nome>.json` | laudo completo: TODAS as plantas, com `bbox`, `saude`, `conf_saude` e `prob_doente` |
| `saida/resumo.csv` | uma linha por foto: total, doentes, sadias, taxa, tempo |

A imagem mostra so as doentes, mas o `.json` tem **todas** as plantas com o
`prob_doente` (0 a 1) — e dele que o backend monta o ranking de suspeitas para
o agronomo revisar, inclusive as de baixa confianca.

## O botao da triagem: `--limiar`

Controla a partir de que `prob_doente` uma planta vira caixa vermelha. **So muda
o desenho**, nao reclassifica o laudo (o JSON sai igual sempre).

```bash
python teste_saude/testar.py --limiar 0.3   # mostra mais suspeitas (recall alto)
python teste_saude/testar.py --limiar 0.7   # so as mais certas (menos falso alarme)
```

- **Limiar baixo (0.2-0.3):** pega quase toda doente, mas marca mais sadias por
  engano. Bom quando o agronomo vai revisar uma a uma e nao quer perder nenhuma.
- **Limiar alto (0.7+):** so o que o modelo tem certeza. Menos ruido, mas pode
  deixar passar doente sutil.

O padrao e **0,5** (a decisao natural do modelo).

## Outras opcoes

```bash
python teste_saude/testar.py --conf 0.20     # detecta mais plantas (infla contagem)
python teste_saude/testar.py --device 0      # usar GPU (se houver)
python teste_saude/testar.py --refazer       # reprocessar o que ja tem saida
```

Fotos que ja tem saida sao **puladas** por padrao. Cada foto leva ~7-16s no CPU
(depende de quantas plantas tem).

## O que olhar no resultado

1. As caixas **vermelhas** estao em plantas que realmente parecem doentes?
2. Tem doente obvia que **nao** foi marcada? (abaixe o `--limiar` e rode de novo
   com `--refazer` para ver se ela aparece com score menor)

## Observacao

Usa o mesmo `src/pipeline.py` que o backend vai importar — o que voce ve aqui e
exatamente o que o produto final produz. Pesos: detector `best_v5.pt`,
classificador `runs/classify/agave_saude_v3_final/weights/best.pt`.
