# -*- coding: utf-8 -*-
"""
Preço diário de cada ação e do Ibovespa pelo Yahoo, e o CDI pelo Banco Central.

Uma chamada por ticker traz o histórico inteiro, ajustado por provento e
desdobramento, e toda atualização baixa de novo o histórico inteiro: assim um
ajuste retroativo (dividendo novo) nunca fica pela metade.

Ticker que saiu da bolsa não existe mais no Yahoo. Para esses o preço vem do
arquivo congelado `dados/deslistadas.csv` (`config.CONGELADO`).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

import config
from conexao import baixar

PASTA = config.DADOS / "precos"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
URL_CDI = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"


def _yahoo(ticker: str) -> pd.DataFrame | None:
    simbolo = ticker if ticker.startswith("^") else f"{ticker}.SA"
    inicio = int(pd.Timestamp(config.INICIO).timestamp()) - 86400 * 400  # folga para as médias
    fim = int(datetime.now(timezone.utc).timestamp())
    r = baixar(URL.format(simbolo), params={"period1": inicio, "period2": fim,
                                             "interval": "1d", "events": "div,splits"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "data": pd.to_datetime(res["timestamp"], unit="s", utc=True)
                  .tz_convert(res["meta"]["exchangeTimezoneName"]).tz_localize(None).normalize(),
        "abertura": q["open"], "maxima": q["high"], "minima": q["low"],
        "fechamento": q["close"], "volume": q["volume"],
        "fechamento_ajustado": (res["indicators"].get("adjclose") or [{"adjclose": q["close"]}])[0]["adjclose"],
    })
    df = df.dropna(subset=["fechamento", "fechamento_ajustado"])

    # Com o pregão aberto, a última linha é o dia em andamento. Fora.
    agora = pd.Timestamp.now(tz="America/Sao_Paulo")
    if agora.hour < 19 and len(df) and df["data"].iloc[-1] == agora.normalize().tz_localize(None):
        df = df.iloc[:-1]
    return df.drop_duplicates("data", keep="last")


def _congelado(ticker: str) -> pd.DataFrame | None:
    if not config.CONGELADO.exists():
        return None
    df = pd.read_csv(config.CONGELADO, parse_dates=["data"])
    df = df[df["ticker"] == ticker].drop(columns="ticker")
    return df if len(df) else None


def _cdi() -> pd.DataFrame:
    """Série 12 do SGS. A API aceita no máximo 10 anos por chamada."""
    partes = []
    ano = pd.Timestamp(config.INICIO).year - 2
    while ano <= pd.Timestamp.today().year:
        r = baixar(URL_CDI, params={"formato": "json", "dataInicial": f"01/01/{ano}",
                                    "dataFinal": f"31/12/{ano + 4}"})
        r.raise_for_status()
        partes.append(pd.DataFrame(r.json()))
        ano += 5
    cdi = pd.concat(partes)
    cdi["data"] = pd.to_datetime(cdi["data"], format="%d/%m/%Y")
    cdi["cdi"] = cdi["valor"].astype(float) / 100
    return cdi[["data", "cdi"]].drop_duplicates("data")


def atualizar(tickers: list[str] | None = None) -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    for t in tickers or [*config.TICKERS, config.REFERENCIA]:
        df, fonte = _yahoo(t), "yahoo"
        if df is None or len(df) < 50:
            df, fonte = _congelado(t), "congelado"
        if df is None:
            print(f"  preço {t:<7} sem fonte, fica fora")
            continue
        df.to_csv(PASTA / f"{t}.csv", index=False)
        print(f"  preço {t:<7} {len(df):>5} pregões até {df['data'].iloc[-1].date()} ({fonte})")
        time.sleep(1)       # o Yahoo devolve 429 para quem pede rápido demais
    cdi = _cdi()
    cdi.to_csv(PASTA / "CDI.csv", index=False)
    print(f"  CDI     {len(cdi):>5} dias até {cdi['data'].iloc[-1].date()}")


def carregar(ticker: str) -> pd.DataFrame:
    arquivo = PASTA / f"{ticker}.csv"
    if not arquivo.exists():
        raise SystemExit(f"sem preço de {ticker}. Rode antes: python baixar.py")
    df = pd.read_csv(arquivo, parse_dates=["data"])
    # Nada depois do fim do episódio: dado novo baixado depois não muda episódio publicado.
    return df[df["data"] <= config.FIM]


def disponiveis() -> list[str]:
    return [t for t in config.TICKERS if (PASTA / f"{t}.csv").exists()]
