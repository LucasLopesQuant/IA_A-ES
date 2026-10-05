# -*- coding: utf-8 -*-
"""
A previsão para o próximo pregão, ação por ação.

    python prever.py               # com os dados que já estão em dados/
    python prever.py --atualizar   # baixa o que saiu de novo antes de prever

Usa a rede final do episódio (`resultados/ep<n>/modelo_final.pt`). Essa rede
treinou só até o corte da validação: é a mesma que perdeu para o sorteio no
backtest. Isto aqui é demonstração do sistema funcionando, não recomendação de compra.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

import config
from baixar import atualizar_tudo
from experimento import escalar
from features import montar, fins_de_janela
from modelo import Rede, prever

PASSADAS = 100


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodio", type=int, default=config.ATUAL)  # lido pelo config.py
    ap.add_argument("--atualizar", action="store_true")
    args = ap.parse_args()

    pesos = config.RESULTADOS / "modelo_final.pt"
    if not pesos.exists():
        raise SystemExit("sem rede treinada. Rode antes: python experimento.py --etapas 5 --episodio N")
    if args.atualizar:
        atualizar_tudo()

    # A previsão é para amanhã: usa o dado até hoje, não só até o fim do episódio.
    config.FIM = "2100-01-01"
    norm = json.loads((config.RESULTADOS / "normalizacao.json").read_text(encoding="utf-8"))
    rede = Rede(len(norm["colunas"]))
    rede.load_state_dict(torch.load(pesos))

    tab = montar()
    fins = fins_de_janela(tab, com_alvo=False)
    ultimos = pd.Series(fins, index=tab["ticker"].to_numpy()[fins]).groupby(level=0).max()
    # Quem não negociou nos últimos 10 dias saiu da bolsa ou está suspensa: não se prevê.
    datas = tab["data"].to_numpy()[ultimos.to_numpy()]
    vivas = datas >= datas.max() - np.timedelta64(10, "D")
    if (~vivas).any():
        print(f"  fora, sem pregão recente: {list(ultimos.index[~vivas])}")
    ultimos = ultimos[vivas]
    prob, incerteza = prever(rede, escalar(tab, norm), ultimos.to_numpy(), passadas=PASSADAS)

    saida = pd.DataFrame({
        "ticker": ultimos.index,
        "dados_ate": tab["data"].to_numpy()[ultimos.to_numpy()],
        "prob_subir": prob, "incerteza": incerteza,
    }).sort_values("prob_subir", ascending=False).reset_index(drop=True)
    saida["sinal"] = np.where(saida["prob_subir"] >= config.LIMIAR, "comprar", "manter")
    saida["top3"] = saida.index < config.TOP

    data = pd.Timestamp(saida["dados_ate"].max()).date()
    saida.to_csv(config.RESULTADOS / f"previsao_{data}.csv", index=False)
    print(f"\nprevisão para o pregão depois de {data} (horizonte de {config.HORIZONTE} dia)\n")
    for _, r in saida.iterrows():
        marca = "  <- top 3" if r["top3"] else ""
        print(f"  {r['ticker']:<6} {r['prob_subir']:6.1%}  ± {r['incerteza']:.1%}  "
              f"{r['sinal']:<8}{marca}")
    atrasadas = saida[saida["dados_ate"] < saida["dados_ate"].max()]
    if len(atrasadas):
        print(f"\n  atenção: {list(atrasadas['ticker'])} com dado mais velho que as outras")


if __name__ == "__main__":
    main()
