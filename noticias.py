# -*- coding: utf-8 -*-
"""
Quantidade de notícias e nota de sentimento por ação.

Não existe na B3 uma API aberta de notícia com nota, como a Alpha Vantage lá fora.
O que existe, oficial e com histórico desde 2003, é o que a própria empresa publica
na CVM: fato relevante e comunicado ao mercado. É a notícia de primeira mão, antes
de virar matéria de jornal.

A nota vem do FinBERT-PT-BR, um BERT treinado em notícia financeira brasileira,
lendo o assunto de cada documento ("Aquisição da empresa X", "Pedido de
recuperação judicial"). Nota = probabilidade de positivo menos a de negativo, de
-1 a +1. Cada texto é lido uma vez só: as notas ficam em `dados/notas_finbert.csv`.

Na primeira vez o modelo (440 MB) desce do Hugging Face para o cache local.
"""
from __future__ import annotations

import pandas as pd

import config
import cvm

ARQUIVO = config.DADOS / "noticias.csv"
NOTAS = config.DADOS / "notas_finbert.csv"
CATEGORIAS = ["Fato Relevante", "Comunicado ao Mercado"]


def _ano(ano: int, cnpjs: set[str]) -> pd.DataFrame | None:
    df = cvm.ler_ano("IPE", "ipe_cia_aberta", ano, "ipe_cia_aberta_{ano}.csv",
                     ["CNPJ_Companhia", "Categoria", "Tipo", "Assunto", "Data_Entrega",
                      "Protocolo_Entrega"])
    if df is None:
        return None
    df = df[df["CNPJ_Companhia"].isin(cnpjs) & df["Categoria"].isin(CATEGORIAS)]
    # Reapresentação do mesmo documento não é notícia nova: fica a primeira entrega.
    df = df.sort_values("Data_Entrega").drop_duplicates("Protocolo_Entrega")
    texto = df["Assunto"].fillna(df["Tipo"]).fillna(df["Categoria"]).str.strip()
    return pd.DataFrame({"cnpj": df["CNPJ_Companhia"], "entrega": df["Data_Entrega"],
                         "categoria": df["Categoria"], "texto": texto})


def _pontuar(textos: list[str]) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  FinBERT-PT-BR lendo {len(textos)} textos novos...")
    tok = AutoTokenizer.from_pretrained(config.FINBERT)
    modelo = AutoModelForSequenceClassification.from_pretrained(config.FINBERT).eval()
    rotulos = {v: int(k) for k, v in modelo.config.id2label.items()}
    notas = []
    with torch.no_grad():
        for i in range(0, len(textos), 64):
            lote = tok(textos[i:i + 64], padding=True, truncation=True, max_length=64,
                       return_tensors="pt")
            p = torch.softmax(modelo(**lote).logits, -1)
            notas.extend((p[:, rotulos["POSITIVE"]] - p[:, rotulos["NEGATIVE"]]).tolist())
            if i % 3200 == 0 and i:
                print(f"    {i}/{len(textos)}")
    return pd.DataFrame({"texto": textos, "nota": notas})


def atualizar() -> None:
    cnpjs = set(cvm.mapa().values())
    velho = pd.read_csv(ARQUIVO, dtype=str) if ARQUIVO.exists() else None
    feitos = set(velho["entrega"].str[:4].astype(int)) if velho is not None else set()
    refazer = {pd.Timestamp.today().year}
    partes = [] if velho is None else [velho[~velho["entrega"].str[:4].astype(int).isin(refazer)]]
    for ano in cvm.anos_desde(pd.Timestamp(config.INICIO).year - 1):
        if ano in feitos and ano not in refazer:
            continue
        df = _ano(ano, cnpjs)
        if df is None:
            continue
        partes.append(df)
        print(f"  notícias {ano}: {len(df):>5} fatos relevantes e comunicados")
    tudo = pd.concat(partes)
    tudo.to_csv(ARQUIVO, index=False)

    notas = pd.read_csv(NOTAS) if NOTAS.exists() else pd.DataFrame(columns=["texto", "nota"])
    novos = sorted(set(tudo["texto"].dropna()) - set(notas["texto"]))
    if novos:
        notas = pd.concat([notas, _pontuar(novos)])
        notas.to_csv(NOTAS, index=False)
    print(f"  notas: {len(notas)} textos diferentes com nota")


def carregar() -> pd.DataFrame:
    if not ARQUIVO.exists():
        print("  aviso: sem notícias (rode python baixar.py). Entram como zero.")
        return pd.DataFrame(columns=["cnpj", "entrega", "categoria", "texto", "nota"])
    df = pd.read_csv(ARQUIVO, parse_dates=["entrega"])
    notas = pd.read_csv(NOTAS) if NOTAS.exists() else pd.DataFrame(columns=["texto", "nota"])
    return df.merge(notas.drop_duplicates("texto"), on="texto", how="left").fillna({"nota": 0.0})


def diario(df: pd.DataFrame, cnpj: str | None, pregoes: pd.DatetimeIndex) -> pd.DataFrame:
    """Por pregão: quantos documentos, quantos fatos relevantes e a nota média."""
    saida = pd.DataFrame(0.0, index=pregoes, columns=["n_noticias", "n_fatos", "sentimento"])
    d = df[df["cnpj"] == cnpj]
    if d.empty:
        return saida
    d = d.assign(pregao=cvm.proximo_pregao(d["entrega"], pregoes)).dropna(subset=["pregao"])
    g = d.groupby("pregao")
    saida["n_noticias"] = g.size().reindex(pregoes, fill_value=0).astype(float)
    saida["n_fatos"] = (d[d["categoria"] == "Fato Relevante"].groupby("pregao").size()
                        .reindex(pregoes, fill_value=0).astype(float))
    saida["sentimento"] = g["nota"].mean().reindex(pregoes).fillna(0.0)
    return saida
