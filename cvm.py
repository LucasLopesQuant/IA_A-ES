# -*- coding: utf-8 -*-
"""
O portal de dados abertos da CVM: um zip por ano, CSV separado por ponto e vírgula,
em latin-1. Daqui saem o insider (VLMO), as notícias (IPE) e o mapa de ticker para
CNPJ (FCA), que é como a CVM identifica a empresa.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

import config
from conexao import baixar

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{nome}_{ano}.zip"
MAPA = config.DADOS / "ticker_cnpj.csv"
# O cadastro da CVM tem código de negociação errado para estas duas ("000000" e "4030").
# CNPJ conferido pelo nome no próprio cadastro.
MANUAL = {"BPAC11": "30.306.294/0001-45", "CSNA3": "33.042.730/0001-04"}


def ler_ano(doc: str, nome: str, ano: int, arquivo: str,
            colunas: list[str] | None = None) -> pd.DataFrame | None:
    """Um CSV de dentro do zip anual. None se o ano ainda não existe."""
    r = baixar(BASE.format(doc=doc, nome=nome, ano=ano))
    if r.status_code == 404:
        return None
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return pd.read_csv(z.open(arquivo.format(ano=ano)), sep=";", encoding="latin-1",
                       dtype=str, usecols=colunas, on_bad_lines="skip")


def anos_desde(inicio: int) -> list[int]:
    return list(range(inicio, pd.Timestamp.today().year + 1))


def atualizar_mapa() -> None:
    """Ticker -> CNPJ pelo Formulário Cadastral. PETR3 e PETR4 caem no mesmo CNPJ."""
    partes = []
    for ano in anos_desde(2018):
        df = ler_ano("FCA", "fca_cia_aberta", ano, "fca_cia_aberta_valor_mobiliario_{ano}.csv",
                     ["CNPJ_Companhia", "Codigo_Negociacao"])
        if df is not None:
            partes.append(df.dropna())
    fca = pd.concat(partes).drop_duplicates()
    fca["codigo"] = fca["Codigo_Negociacao"].str.strip().str.upper()
    fca["raiz"] = fca["codigo"].str[:4]

    linhas = []
    for t in config.TICKERS:
        if t in MANUAL:
            linhas.append({"ticker": t, "cnpj": MANUAL[t]})
            continue
        achou = fca[fca["codigo"] == t]
        if achou.empty:                       # unit ou código antigo: casa pela raiz
            achou = fca[fca["raiz"] == t[:4]]
        if achou.empty:
            print(f"  {t}: sem CNPJ na CVM, fica sem insider e sem notícia")
            continue
        linhas.append({"ticker": t, "cnpj": achou["CNPJ_Companhia"].mode().iloc[0]})
    config.DADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(MAPA, index=False)
    print(f"  mapa: {len(linhas)} de {len(config.TICKERS)} tickers com CNPJ")


def mapa() -> dict[str, str]:
    if not MAPA.exists():
        atualizar_mapa()
    return pd.read_csv(MAPA, dtype=str).set_index("ticker")["cnpj"].to_dict()


def proximo_pregao(datas: pd.Series, pregoes: pd.DatetimeIndex) -> pd.Series:
    """O pregão seguinte à data. A CVM não diz a hora da entrega, então documento
    entregue hoje só conta amanhã: pode ter saído às 22h."""
    pos = pregoes.searchsorted(datas.to_numpy(), side="right")
    saida = pd.Series(pd.NaT, index=datas.index)
    ok = pos < len(pregoes)
    saida[ok] = pregoes[pos[ok]]
    return saida
