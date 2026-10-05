# -*- coding: utf-8 -*-
"""
Compra e venda de diretor, conselheiro e controlador, pela CVM.

Todo mês a empresa entrega à CVM o formulário de "valores mobiliários negociados e
detidos" (art. 11 da antiga Instrução 358, hoje Resolução 44): quem da administração
comprou ou vendeu ação da própria empresa, quanto e a que preço. É o Formulário 4 da
SEC em versão brasileira, com uma diferença que importa: sai uma vez por mês, até o
dia 10 do mês seguinte, e não em dois dias. O mercado fica sabendo com atraso.

O dado aberto começa em 2018. Antes disso as colunas entram zeradas, e
`insider_coberto` avisa a rede de que o zero ali é "não sei", não "ninguém mexeu".
"""
from __future__ import annotations

import pandas as pd

import config
import cvm

ARQUIVO = config.DADOS / "insiders.csv"
DESDE = 2018
LADOS = {"Compra à vista": "compra", "Venda à vista": "venda"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")


def _ano(ano: int, cnpjs: set[str]) -> pd.DataFrame | None:
    indice = cvm.ler_ano("VLMO", "vlmo_cia_aberta", ano, "vlmo_cia_aberta_{ano}.csv",
                         ["CNPJ_Companhia", "Data_Referencia", "Versao", "Data_Entrega"])
    if indice is None:
        return None
    mov = cvm.ler_ano("VLMO", "vlmo_cia_aberta", ano, "vlmo_cia_aberta_con_{ano}.csv",
                      ["CNPJ_Companhia", "Data_Referencia", "Versao", "Tipo_Empresa",
                       "Tipo_Cargo", "Tipo_Movimentacao", "Tipo_Ativo", "Volume"])
    mov = mov[mov["CNPJ_Companhia"].isin(cnpjs)
              & mov["Tipo_Movimentacao"].isin(LADOS)
              & mov["Tipo_Ativo"].isin(["Ações", "Units"])
              & (mov["Tipo_Empresa"] == "Companhia")]
    # Reapresentação substitui a original. Fica a última versão, com a data de entrega
    # dela: é conservador, a informação só conta a partir de quando a versão final saiu.
    mov = mov.assign(v=pd.to_numeric(mov["Versao"]))
    ultima = mov.groupby(["CNPJ_Companhia", "Data_Referencia"])["v"].transform("max")
    mov = mov[mov["v"] == ultima]
    df = mov.merge(indice, on=["CNPJ_Companhia", "Data_Referencia", "Versao"])
    return pd.DataFrame({
        "cnpj": df["CNPJ_Companhia"], "referencia": df["Data_Referencia"],
        "entrega": df["Data_Entrega"], "cargo": df["Tipo_Cargo"],
        "lado": df["Tipo_Movimentacao"].map(LADOS), "volume": _num(df["Volume"]).fillna(0),
    })


def atualizar() -> None:
    cnpjs = set(cvm.mapa().values())
    velho = pd.read_csv(ARQUIVO, dtype=str) if ARQUIVO.exists() else None
    # Ano fechado não muda; o atual e o anterior (reapresentações) baixam de novo.
    feitos = set(velho["referencia"].str[:4].astype(int)) if velho is not None else set()
    refazer = {pd.Timestamp.today().year, pd.Timestamp.today().year - 1}
    partes = [] if velho is None else [velho[~velho["referencia"].str[:4].astype(int).isin(refazer)]]
    for ano in cvm.anos_desde(DESDE):
        if ano in feitos and ano not in refazer:
            continue
        df = _ano(ano, cnpjs)
        if df is None:
            continue
        partes.append(df)
        print(f"  insider {ano}: {len(df):>5} negociações das nossas empresas")
    pd.concat(partes).to_csv(ARQUIVO, index=False)


def carregar() -> pd.DataFrame:
    if not ARQUIVO.exists():
        print("  aviso: sem dados de insider (rode python baixar.py). Entram como zero.")
        return pd.DataFrame(columns=["cnpj", "referencia", "entrega", "cargo", "lado", "volume"])
    return pd.read_csv(ARQUIVO, parse_dates=["referencia", "entrega"]).drop_duplicates()


def diario(df: pd.DataFrame, cnpj: str | None, pregoes: pd.DatetimeIndex) -> pd.DataFrame:
    """Por pregão: quantas negociações de compra e de venda o mercado ficou sabendo
    naquele dia, e o volume de cada lado."""
    saida = pd.DataFrame(0.0, index=pregoes,
                         columns=["n_compra", "n_venda", "valor_compra", "valor_venda"])
    # O arquivo aberto de 2018 só traz o fim do ano: a cobertura começa na primeira entrega.
    inicio = df["entrega"].min() if len(df) else pd.Timestamp.max
    saida["coberto"] = (pregoes > inicio).astype(float)
    d = df[df["cnpj"] == cnpj]
    if d.empty:
        return saida
    d = d.assign(pregao=cvm.proximo_pregao(d["entrega"], pregoes)).dropna(subset=["pregao"])
    for lado in ("compra", "venda"):
        g = d[d["lado"] == lado].groupby("pregao")["volume"]
        saida[f"n_{lado}"] = g.size().reindex(pregoes, fill_value=0).astype(float)
        saida[f"valor_{lado}"] = g.sum().reindex(pregoes, fill_value=0.0)
    return saida
