# -*- coding: utf-8 -*-
"""
Do dado bruto até a janela que entra na rede.

Carregar, limpar, calcular retorno, criar os alvos (sobe ou não em 1, 7, 30 e 180
dias), e depois, no `experimento.py`, normalizar com a média do treino e recortar
as janelas deslizantes.

Tudo que vira feature é relativo (preço sobre a média, volume sobre a média), para
uma ação de R$ 2 e uma de R$ 80 falarem a mesma língua dentro da rede.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import cvm
import insiders
import noticias
import precos

RETORNO_ABSURDO = 0.6   # acima disso num dia é erro de cadastro, não mercado

FEATURES_BASE = [
    # preço
    "retorno", "mm10", "mm20", "mm30", "mme10", "mme20", "mme50",
    "volatilidade", "amplitude", "volume", "rsi14",
    # mercado
    "ibov", "ibov_mm20",
    # calendário
    "dia_mes", "dia_semana", "mes",
    # insider (CVM)
    "insider_compras_90d", "insider_vendas_90d", "insider_saldo_90d",
    "insider_comprando", "insider_coberto",
    # notícias (CVM + FinBERT-PT-BR)
    "n_noticias", "noticias_30d", "fatos_30d", "sentimento", "sentimento_5d",
]
# As 26 do episódio 1 e as que o episódio escolhido acrescenta (config.EXTRAS).
FEATURES = FEATURES_BASE + list(config.EXTRAS)

# As duas colunas que só existem para os testes de sanidade. Nunca entram por padrão.
RESPOSTA = "alvo"               # a própria resposta
FECH_AMANHA = "fech_amanha"     # o fechamento de amanhã, a pista indireta


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Amplitude média verdadeira, sobre o preço: quanto a ação costuma andar num dia,
    contando o salto do fechamento de ontem para a abertura de hoje."""
    fech_ontem = df["fechamento"].shift(1)
    verdadeira = pd.concat([df["maxima"] - df["minima"], (df["maxima"] - fech_ontem).abs(),
                            (df["minima"] - fech_ontem).abs()], axis=1).max(axis=1)
    return verdadeira.ewm(alpha=1 / n, adjust=False).mean() / df["fechamento"]


def _super_smoother(x: np.ndarray, periodo: float) -> np.ndarray:
    """Filtro de dois polos do John Ehlers: suaviza quase como uma média móvel, com bem
    menos atraso. Só olha para trás (recursivo), então não vaza futuro."""
    a = np.exp(-1.414 * np.pi / periodo)
    c2 = 2 * a * np.cos(1.414 * np.pi / periodo)
    c3 = -a * a
    c1 = 1 - c2 - c3
    y = x.astype(float).copy()
    for i in range(2, len(x)):
        y[i] = c1 * (x[i] + x[i - 1]) / 2 + c2 * y[i - 1] + c3 * y[i - 2]
    return y


def _roofing(x: np.ndarray, longo: float = 48, curto: float = 10) -> np.ndarray:
    """Roofing Filter do Ehlers: um passa-alta tira a tendência mais longa que `longo`
    pregões, e o Super Smoother tira o chiado mais curto que `curto`. Sobram os ciclos."""
    ang = 0.707 * 2 * np.pi / longo
    alfa = (np.cos(ang) + np.sin(ang) - 1) / np.cos(ang)
    hp = np.zeros(len(x))
    for i in range(2, len(x)):
        hp[i] = ((1 - alfa / 2) ** 2 * (x[i] - 2 * x[i - 1] + x[i - 2])
                 + 2 * (1 - alfa) * hp[i - 1] - (1 - alfa) ** 2 * hp[i - 2])
    return _super_smoother(hp, curto)


def _rsi(preco: pd.Series, n: int = 14) -> pd.Series:
    d = preco.diff()
    alta = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    baixa = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 1 - 1 / (1 + alta / baixa) - 0.5


def _por_acao(ticker: str, cnpj: str | None, ibov: pd.DataFrame,
              ins: pd.DataFrame, noti: pd.DataFrame) -> pd.DataFrame:
    df = precos.carregar(ticker).set_index("data").sort_index()
    df = df[df["volume"] > 0]
    preco = df["fechamento_ajustado"]
    pregoes = df.index

    f = pd.DataFrame(index=pregoes)
    f["fechamento"] = preco
    f["retorno"] = preco.pct_change()
    mm20 = preco.rolling(20).mean()
    for n in (10, 20, 30):
        f[f"mm{n}"] = preco / preco.rolling(n).mean() - 1
    for n in (10, 20, 50):
        f[f"mme{n}"] = preco / preco.ewm(span=n, adjust=False).mean() - 1
    f["volatilidade"] = f["retorno"].rolling(20).std()
    f["amplitude"] = (df["maxima"] - df["minima"]) / df["fechamento"]
    f["volume"] = np.log(df["volume"] / df["volume"].rolling(20).mean())
    f["rsi14"] = _rsi(preco)

    # Episódio 2, sugestões do @leonardomotta150. Sobre o log do preço, para não depender
    # do nível (R$ 2 ou R$ 80): o desvio do preço para a versão suave, e o ciclo que sobra.
    f["atr14"] = _atr(df)
    # O Yahoo tem preço ajustado negativo em 2006-2007 para ELPL4 e OIBR4 (erro de ajuste).
    # Filtro recursivo carregaria esse buraco para sempre: ele começa depois do último
    # preço inválido, e antes disso fica vazio.
    log_preco = np.log(preco.where(preco > 0)).to_numpy()
    inicio = np.flatnonzero(~np.isfinite(log_preco))
    inicio = inicio[-1] + 1 if len(inicio) else 0
    ss, roof = np.full(len(log_preco), np.nan), np.full(len(log_preco), np.nan)
    if len(log_preco) - inicio > 50:
        ss[inicio:] = log_preco[inicio:] - _super_smoother(log_preco[inicio:], 10)
        roof[inicio:] = _roofing(log_preco[inicio:])
    f["ehlers_ss"] = ss
    f["ehlers_roofing"] = roof

    f["ibov"] = ibov["retorno"].reindex(pregoes)
    f["ibov_mm20"] = ibov["mm20"].reindex(pregoes)

    f["dia_mes"] = pregoes.day / 31
    f["dia_semana"] = pregoes.dayofweek / 4
    f["mes"] = pregoes.month / 12

    i = insiders.diario(ins, cnpj, pregoes)
    f["insider_compras_90d"] = np.log1p(i["n_compra"].rolling(63, min_periods=1).sum())
    f["insider_vendas_90d"] = np.log1p(i["n_venda"].rolling(63, min_periods=1).sum())
    compra = i["valor_compra"].rolling(63, min_periods=1).sum()
    venda = i["valor_venda"].rolling(63, min_periods=1).sum()
    f["insider_saldo_90d"] = ((compra - venda) / (compra + venda)).fillna(0)
    f["insider_comprando"] = (i["n_compra"].rolling(21, min_periods=1).sum() > 0).astype(float)
    f["insider_coberto"] = i["coberto"]

    n = noticias.diario(noti, cnpj, pregoes)
    f["n_noticias"] = np.log1p(n["n_noticias"])
    f["noticias_30d"] = np.log1p(n["n_noticias"].rolling(21, min_periods=1).sum())
    f["fatos_30d"] = np.log1p(n["n_fatos"].rolling(21, min_periods=1).sum())
    f["sentimento"] = n["sentimento"]
    f["sentimento_5d"] = n["sentimento"].rolling(5, min_periods=1).mean()

    # Os alvos, um por horizonte. São as únicas colunas que olham para o futuro.
    for h in config.HORIZONTES:
        f[f"ret_{h}d"] = preco.shift(-h) / preco - 1
        f[f"alvo_{h}d"] = (f[f"ret_{h}d"] > 0).astype(float).where(f[f"ret_{h}d"].notna())
    if config.ALVO == "retorno":
        # Episódio 2: quanto, e não só se. Retorno em log até o fim do horizonte.
        f[RESPOSTA] = np.log(preco.shift(-config.HORIZONTE) / preco)
    else:
        f[RESPOSTA] = f[f"alvo_{config.HORIZONTE}d"]
    f["ret_amanha"] = f["ret_1d"]
    f[FECH_AMANHA] = preco.shift(-1) / mm20 - 1     # compara com o mm20 de hoje e acha a resposta

    # Só entra na carteira quem negocia de verdade: R$ 2 milhões por dia, mediana de 20.
    financeiro = (df["fechamento"] * df["volume"]).rolling(20).median()
    f["negociavel"] = financeiro > config.LIQUIDEZ_MIN
    return f


def _ibov() -> pd.DataFrame:
    p = precos.carregar(config.REFERENCIA).set_index("data")["fechamento_ajustado"]
    return pd.DataFrame({"retorno": p.pct_change(), "mm20": p / p.rolling(20).mean() - 1})


def cdi() -> pd.Series:
    df = precos.carregar("CDI").set_index("data")["cdi"]
    return df.sort_index()


def montar(tickers: list[str] | None = None) -> pd.DataFrame:
    """Tabela longa (ticker, data), ordenada por ticker e data, sem buraco nas features."""
    ibov, ins, noti, mapa = _ibov(), insiders.carregar(), noticias.carregar(), cvm.mapa()
    partes, cortados = [], 0
    for t in tickers or precos.disponiveis():
        f = _por_acao(t, mapa.get(t), ibov, ins, noti)
        absurdo = f["retorno"].abs() > RETORNO_ABSURDO
        cortados += int(absurdo.sum())
        # Sai o dia do erro, o anterior (cujo alvo é o salto falso) e o seguinte (que desfaz o salto).
        fora = absurdo | absurdo.shift(-1, fill_value=False) | absurdo.shift(1, fill_value=False)
        partes.append(f[~fora].assign(ticker=t))
    if cortados:
        print(f"  {cortados} dias com retorno acima de {RETORNO_ABSURDO:.0%} tratados como erro de dado")

    tab = pd.concat(partes).rename_axis("data").reset_index()
    tab = tab[tab["data"] >= config.INICIO]
    tab = tab.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return tab.sort_values(["ticker", "data"]).reset_index(drop=True)


def fins_de_janela(tab: pd.DataFrame, com_alvo: bool = True) -> np.ndarray:
    """Linhas com JANELA pregões da mesma ação atrás de si. Cada índice é o último dia.

    A janela da linha i é tab[i - JANELA + 1 : i + 1]. Com `com_alvo`, só as que já
    têm resposta (o futuro do horizonte já aconteceu).
    """
    ok = tab["ticker"].eq(tab["ticker"].shift(config.JANELA - 1)).to_numpy()
    ok[: config.JANELA - 1] = False
    if com_alvo:
        ok &= tab[RESPOSTA].notna().to_numpy()
    return np.flatnonzero(ok)


def janelas(matriz: np.ndarray, fins: np.ndarray) -> np.ndarray:
    """Recorta as janelas de uma vez: (n, JANELA, n_features)."""
    passos = np.arange(-config.JANELA + 1, 1)
    return matriz[fins[:, None] + passos]
