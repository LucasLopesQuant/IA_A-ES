# IA que tenta prever a bolsa brasileira (versão 2)

Uma LSTM que olha os últimos 30 pregões de 61 ações da B3 (preço, médias, compra e venda de
diretores na CVM e notícias da empresa com nota de sentimento) e tenta escolher as três
melhores para os próximos dias. O placar é sempre o mesmo: a IA contra o Ibovespa comprado e
segurado, o CDI e 1.000 pessoas escolhendo ação no sorteio, com e sem custo de operação.

Este é o código do **episódio 2** da série no YouTube, a versão com as sugestões dos
comentários. O episódio 1 continua reproduzível no mesmo código (`--episodio 1`).

**Não é recomendação de investimento.** A IA perde para o sorteio nas duas versões, como o
placar abaixo mostra.

## O que mudou na versão 2

| mudança | de quem | onde no código |
|---|---|---|
| horizonte de 5 dias em vez de 1 (a carteira fica 5 dias parada) | sugestão do @leonardomotta150 | `config.py`, `EPISODIOS[2]` |
| ATR de 14 dias como informação nova | sugestão do @leonardomotta150 | `features.py`, `atr14` |
| filtros do John Ehlers: Super Smoother e Roofing Filter | sugestão do @leonardomotta150 | `features.py`, `ehlers_ss` e `ehlers_roofing` |
| o alvo vira o retorno em 5 dias (regressão), e não mais "sobe ou não sobe" | nossa | `config.py`, `alvo: "retorno"` |
| o dado recente pesa mais no treino (meia-vida de 3 anos) | nossa | `experimento.py` e `modelo.py`, `pesos` |
| callback ligado na rede final: o treino para quando a validação para de melhorar | nossa | `config.py`, `final: 4` |
| correção do preço ajustado negativo do Yahoo (ELPL4 e OIBR4 em 2006-2007) | achado no caminho | `features.py` |

A volatilidade de 20 períodos, também sugerida, já estava no modelo desde a versão 1.

## O placar

R$ 1 investido no começo do período de teste (janeiro de 2023 a setembro de 2026) vira:

| versão | IA top 3 | IA com custo | Ibovespa | CDI | giro da carteira por dia |
|---|---|---|---|---|---|
| 1 | 0,31 | 0,09 | 1,68 | 1,57 | 162% |
| 2 | 0,25 | 0,22 | 1,68 | 1,57 | 21% |

O custo caiu de verdade, mas contra 1.000 sorteios de três ações a cada cinco dias a versão 2
fica 3,25 desvios abaixo da média. Olhando dentro da IA (`resultados/ep2/importancia.csv` e
as faixas de previsão), ela acerta a direção no meio das previsões e erra justamente nos 10%
em que mais confia, onde se concentram as ações mais voláteis do período (Americanas, Pão de
Açúcar, Gafisa, Oi). As três informações novas deram importância negativa.

## Como rodar

```bash
pip install -r requirements.txt
python baixar.py                       # preços, CDI, insider e notícias (dado aberto, sem chave)
python experimento.py --episodio 2     # treina e testa; resultados em resultados/ep2/
python prever.py --episodio 2          # previsão do próximo pregão com a rede salva
```

`python experimento.py --rapido` roda poucas épocas por etapa, só para ver se tudo funciona.
Sem `--episodio`, vale o `ATUAL` do `config.py`.

O `baixar.py` demora na primeira vez: as notícias passam pelo FinBERT-PT-BR
(`lucas-leme/FinBERT-PT-BR`, baixado do Hugging Face) para ganhar a nota de sentimento.

## O que vem junto e o que não vem

| caminho | o que é |
|---|---|
| `dados/deslistadas.csv` | o histórico de OGXP3, FIBR3 e ELPL4, que saíram da bolsa e não estão mais no Yahoo. Vai junto porque não dá para baixar de novo |
| `resultados/ep1/`, `resultados/ep2/` | o resumo de cada rodada, as curvas, a importância, os sorteios finais e a rede salva (`modelo_final.pt`), que o `prever.py` usa |
| o resto de `dados/` | não vem: o `baixar.py` refaz |
| as previsões completas e as amostras do sorteio | não vêm (são grandes): o `experimento.py` refaz |

Rodar de novo dá números muito próximos, mas não idênticos: o Yahoo reajusta o preço
histórico a cada provento novo. A data final de cada episódio fica travada no `config.py`.

## Cuidados que mudam o número

- **Nada vê o futuro.** Documento da CVM só conta no pregão seguinte à entrega; o insider é
  mensal e entra a partir da data de entrega.
- **Insider só existe desde 2019** no dado aberto. Antes disso, `insider_coberto` avisa a rede
  que o zero ali quer dizer "não sei".
- **Erro de cadastro.** Dia com mais de 60% de variação sai, junto com o anterior e o seguinte
  (saltos falsos de grupamento no Yahoo).
- **Sobreviventes.** As ações que quebraram ou saíram da bolsa entram no teste. Sem elas, o
  backtest só veria quem sobreviveu.
- **Divisão por tempo, nunca por ação.** Treino até julho de 2021, validação até janeiro de
  2023, teste depois.

## Os arquivos

| arquivo | o que faz |
|---|---|
| `config.py` | todas as decisões que mudam o número, e a configuração de cada episódio |
| `baixar.py` | atualiza todas as fontes |
| `precos.py` | preço pelo Yahoo (ajustado), Ibovespa e CDI pelo Banco Central |
| `cvm.py` | o mapa de ticker para CNPJ, pelo Formulário Cadastral da CVM |
| `insiders.py` | negociação de diretores, conselheiros e controladores (CVM) |
| `noticias.py` | fatos relevantes e comunicados (CVM), com nota do FinBERT-PT-BR |
| `features.py` | as informações de cada pregão e as janelas de 30 dias |
| `modelo.py` | convolução 1D + LSTM + dropout de Monte Carlo |
| `experimento.py` | as etapas de teste, o backtest, o sorteio e a importância por permutação |
| `prever.py` | a previsão do próximo pregão |
| `conexao.py` | rede com retentativa e IPv4 forçado |
