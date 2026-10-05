# -*- coding: utf-8 -*-
"""
Todas as decisões que mudam o número. Nenhum outro arquivo tem constante de experimento.

O sistema do vídeo nº 9 do LosingLoonies, peça por peça, na B3. O que mudou em
relação ao americano:

    S&P 500 comprado e segurado   ->  Ibovespa comprado e segurado, e o CDI
    insider pela SEC              ->  formulário de negociação de administradores da CVM
    notícia pela Alpha Vantage    ->  fato relevante e comunicado ao mercado da CVM,
                                      com a nota do FinBERT-PT-BR
    só empresa viva               ->  entram também as que quebraram ou saíram da bolsa

Cada episódio da série é uma configuração (EPISODIOS, no fim deste arquivo). O código é
um só e evolui; o episódio 1 continua reproduzível porque congela as próprias escolhas e
a data final do dado. Escolher o episódio: `--episodio 2` na linha de comando ou a
variável de ambiente EPISODIO. Sem nada, vale o ATUAL.
"""
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
ATUAL = 2


def _escolher_episodio() -> int:
    if "--episodio" in sys.argv:
        return int(sys.argv[sys.argv.index("--episodio") + 1])
    return int(os.getenv("EPISODIO", ATUAL))


EPISODIO = _escolher_episodio()
RESULTADOS = AQUI / "resultados" / f"ep{EPISODIO}"

# O universo: 51 ações líquidas e 10 que quebraram, saíram
# da bolsa ou quase morreram. Ponha qualquer ticker aqui e rode `python baixar.py`.
TICKERS = [
    "ABEV3", "ASAI3", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BEEF3", "BPAC11",
    "BRKM5", "CMIG4", "COGN3", "CPFE3", "CSAN3", "CSNA3", "CVCB3", "CYRE3", "ENEV3",
    "EQTL3", "GGBR4", "GOAU4", "HAPV3", "ITSA4", "ITUB4", "KLBN11", "LREN3", "MGLU3",
    "MRVE3", "MULT3", "PCAR3", "PETR3", "PETR4", "PRIO3", "QUAL3", "RADL3", "RAIL3",
    "RDOR3", "RENT3", "SBSP3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3", "USIM5",
    "VALE3", "VBBR3", "VIVT3", "WEGE3", "YDUQ3",
    # as que quebraram ou quase: sem elas o teste só vê quem sobreviveu
    "AMER3", "GFSA3", "LIGT3", "OIBR3", "OIBR4", "OSXB3", "PDGR3", "RSID3",
    "OGXP3", "FIBR3", "ELPL4",
]
# O Yahoo não tem mais o histórico de quem saiu da bolsa (OGXP3, FIBR3, ELPL4). Esse vem
# congelado em `dados/deslistadas.csv`, que vai junto com o código.
CONGELADO = DADOS / "deslistadas.csv"

REFERENCIA = "^BVSP"        # o Ibovespa comprado e segurado
INICIO = "2008-01-01"       # 2008 inteiro dentro do treino, que é a lição do vídeo

# Os alvos montados. Qual deles a rede tenta prever é escolha do episódio (lá embaixo).
HORIZONTES = (1, 5, 7, 30, 180)

JANELA = 30                 # dias que a rede enxerga para decidir o seguinte
FRACAO_TESTE = 0.20         # 80/20 por tempo, nunca por ação
FRACAO_VALIDACAO = 0.10     # o fim dos 80% de treino, para os callbacks
LIQUIDEZ_MIN = 2e6          # R$ por dia (mediana de 20 pregões) para a ação poder ser comprada

# A estratégia e a régua.
LIMIAR = 0.7                # acima disso a rede diz "comprar", abaixo "manter"
TOP = 3                     # todo dia, as 3 ações com maior probabilidade de subir
SORTEIOS = 1000             # quantas pessoas escolhendo ação no sorteio todo dia
SORTEIO_POR_DIA = 1         # o vídeo compara com UMA ação sorteada por dia
# Custo por lado: 0,03% de emolumento e liquidação da B3 e 0,05% de spread e escorregão.
# O vídeo não cobra custo; aqui sai com e sem, lado a lado.
CUSTO_LADO = 0.0008

# O treino.
LOTE = 512
TAXA = 1e-3
PASSADAS_MC = 30            # quantas vezes a mesma janela passa com dropout ligado
SEMENTE = 7

# O modelo de sentimento em português, treinado em notícia financeira brasileira.
FINBERT = "lucas-leme/FinBERT-PT-BR"


# ------------------------------------------------------------------ os episódios
#
# horizonte   quantos pregões à frente a rede tenta prever, e quanto tempo a carteira fica
# alvo        "sobe" (1 ou 0, classificação) ou "retorno" (o retorno em log, regressão)
# fim         último pregão usado; congela o episódio mesmo com dado novo baixado depois
# extras      informações a mais além das 26 do episódio 1
# meia_vida   peso do dado recente no treino, em anos (None = todo dia pesa igual)
# final       a etapa do experimento.py que gera a rede do backtest e das previsões
EPISODIOS = {
    # Publicado em 29/09/2026. Não mexer: os números estão no ar.
    1: {"horizonte": 1, "alvo": "sobe", "fim": "2026-09-28", "extras": [], "meia_vida": None,
        "final": 5},
    # As sugestões dos comentários do episódio 1 e as que o LosingLoonies aceitou no nº 14.
    2: {"horizonte": 5, "alvo": "retorno", "fim": "2026-09-28",
        "extras": ["atr14", "ehlers_ss", "ehlers_roofing"], "meia_vida": 3.0,
        # o callback, como o LosingLoonies adotou no nº 14: a rede de 200 épocas decora
        "final": 4},
}
_EP = EPISODIOS[EPISODIO]
HORIZONTE = _EP["horizonte"]
ALVO = _EP["alvo"]
FIM = _EP["fim"]
EXTRAS = _EP["extras"]
MEIA_VIDA = _EP["meia_vida"]
FINAL = _EP["final"]
