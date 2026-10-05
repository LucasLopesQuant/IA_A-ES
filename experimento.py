# -*- coding: utf-8 -*-
"""
O experimento do vídeo, na ordem em que ele acontece, na B3.

    python experimento.py                 # tudo; em CPU leva de 1 a 2 horas
    python experimento.py --rapido        # 1 a 3 épocas por etapa, só para ver se roda
    python experimento.py --etapas 5      # só a etapa final (a que o prever.py usa)

1. A resposta dentro das features (1 época). Acerto alto, sinais de compra, backtest lindo.
2. Tira a resposta (1 época). O acerto tem que cair para perto de 50%.
3. A pista indireta: o fechamento de amanhã dentro da janela (50 épocas). Tem que funcionar.
4. Tudo limpo, 200 épocas com o callback. Ele para cedo.
5. Tudo limpo, 200 épocas sem callback. Esta é a rede do backtest final, contra o
   Ibovespa comprado e segurado, o CDI e 1.000 pessoas comprando ação no sorteio.

As etapas 1 a 3 não são para prever nada: são o teste de sanidade. Se a rede não
acerta nem com a resposta na mão, o problema é o código, não o mercado.

Tudo vai para `resultados/ep<n>/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import config
from features import FEATURES, FECH_AMANHA, RESPOSTA, cdi, fins_de_janela, montar
from modelo import avaliar, prever, treinar

SAIDA = config.RESULTADOS

ETAPAS = {
    # número: (nome, colunas extras, épocas, callback)
    1: ("com_resposta", [RESPOSTA], 1, False),
    2: ("sem_resposta", [], 1, False),
    3: ("fech_amanha", [FECH_AMANHA], 50, False),
    4: ("limpo_callback", [], 200, True),
    5: ("limpo_200", [], 200, False),
}


# ---------------------------------------------------------------- divisão e escala

def dividir(tab: pd.DataFrame, fins: np.ndarray) -> dict:
    """80/20 por tempo, e os últimos 10% do treino como validação.

    Entre um bloco e o seguinte fica um vão de HORIZONTE pregões: o alvo da última
    janela do treino olha para a frente, e sem o vão ele enxergaria o primeiro dia
    da validação.
    """
    datas = tab["data"].to_numpy()[fins]
    unicas = np.unique(datas)
    i_teste = int(len(unicas) * (1 - config.FRACAO_TESTE))
    i_val = int(i_teste * (1 - config.FRACAO_VALIDACAO))
    h = config.HORIZONTE
    return {
        "treino": fins[datas < unicas[i_val - h]],
        "validacao": fins[(datas >= unicas[i_val]) & (datas < unicas[i_teste - h])],
        "teste": fins[datas >= unicas[i_teste]],
        "corte_validacao": pd.Timestamp(unicas[i_val]),
        "corte_teste": pd.Timestamp(unicas[i_teste]),
    }


def normalizacao(tab: pd.DataFrame, colunas: list[str], ate: pd.Timestamp) -> dict:
    """Média e desvio só do treino. Usar a validação ou o teste aqui já é vazamento."""
    treino = tab.loc[tab["data"] < ate, colunas]
    return {"colunas": colunas, "media": treino.mean().tolist(),
            "desvio": treino.std().replace(0, 1).tolist()}


def escalar(tab: pd.DataFrame, norm: dict) -> np.ndarray:
    z = (tab[norm["colunas"]].to_numpy(np.float64) - norm["media"]) / norm["desvio"]
    return np.nan_to_num(np.clip(z, -5, 5)).astype(np.float32)


# ---------------------------------------------------------------- backtest

def _estatisticas(r: pd.Series) -> dict:
    acum = (1 + r).cumprod()
    anos = len(r) / 252
    return {"multiplicador": float(acum.iloc[-1]),
            "retorno_aa": float(acum.iloc[-1] ** (1 / anos) - 1),
            "vol_aa": float(r.std() * np.sqrt(252)),
            "pior_queda": float((acum / acum.cummax() - 1).min())}


def _sorteio(ret_por_dia: list[np.ndarray], rng: np.random.Generator, k: int) -> np.ndarray:
    """SORTEIOS pessoas, cada uma comprando k ações sorteadas todo dia. (dias, pessoas)."""
    saida = np.empty((len(ret_por_dia), config.SORTEIOS))
    for i, ret in enumerate(ret_por_dia):
        escolha = np.argsort(rng.random((config.SORTEIOS, len(ret))), axis=1)[:, :k]
        saida[i] = ret[escolha].mean(1)
    return saida


def _regua(final_modelo: float, finais: np.ndarray) -> dict:
    """Onde o modelo cai na distribuição do sorteio. Tudo no log: dinheiro multiplica."""
    log = np.log(finais)
    return {"media": float(np.exp(log.mean())),
            "faixa_1s": [float(np.exp(log.mean() - log.std())), float(np.exp(log.mean() + log.std()))],
            "faixa_3s": [float(np.exp(log.mean() - 3 * log.std())),
                         float(np.exp(log.mean() + 3 * log.std()))],
            "melhor": float(finais.max()), "pior": float(finais.min()),
            "z_modelo": float((np.log(final_modelo) - log.mean()) / log.std()),
            "percentil_modelo": float((finais < final_modelo).mean())}


def backtest(pred: pd.DataFrame, ibov: pd.Series, taxa_cdi: pd.Series,
             semente: int = 42) -> tuple[pd.DataFrame, dict]:
    """Top 3 por dia, com e sem custo, contra Ibovespa, CDI e o sorteio.

    Só entra no dia quem é negociável (R$ 2 milhões por dia): a IA e o sorteio
    escolhem do mesmo cardápio.
    """
    rng = np.random.default_rng(semente)
    pred = pred[pred["negociavel"]]
    grupos = [g for _, g in pred.groupby("data", sort=True)]
    dias = pd.DatetimeIndex([g["data"].iloc[0] for g in grupos])

    top3, limiar, giro, acertos, n_sinais = [], [], [], [], []
    ontem: set = set()
    for g in grupos:
        g = g.sort_values("prob", ascending=False)
        escolha = g.head(config.TOP)
        carteira = set(escolha["ticker"])
        giro.append(len(carteira ^ ontem) / config.TOP)   # compra nova + venda velha
        ontem = carteira
        top3.append(escolha["ret_amanha"].mean())
        acertos.extend((escolha["ret_amanha"] > 0).tolist())
        sinal = g[g["prob"] >= config.LIMIAR]
        n_sinais.append(len(sinal))
        limiar.append(sinal["ret_amanha"].mean() if len(sinal) else 0.0)

    top3, giro = np.array(top3), np.array(giro)
    diario = pd.DataFrame({
        "modelo_top3": top3,
        "modelo_top3_custo": top3 - giro * config.CUSTO_LADO,
        "modelo_limiar": limiar,
        "ibov": ibov.shift(-1).reindex(dias).fillna(0).to_numpy(),        # o retorno de amanhã
        "cdi": taxa_cdi.shift(-1).reindex(dias).ffill().fillna(0).to_numpy(),
    }, index=dias)
    curvas = (1 + diario).cumprod()

    ret_por_dia = [g["ret_amanha"].to_numpy() for g in grupos]
    resumo = {"dias": len(dias), "inicio": str(dias[0].date()), "fim": str(dias[-1].date()),
              "acoes_por_dia": float(np.mean([len(r) for r in ret_por_dia])),
              "curvas": {c: _estatisticas(diario[c]) for c in diario},
              "acerto_top3": float(np.mean(acertos)),
              "acerto_uma_qualquer": float(pred["alvo"].mean()),
              "dias_com_sinal": int(np.sum(np.array(n_sinais) > 0)),
              "sinais_total": int(np.sum(n_sinais)),
              "giro_medio_dia": float(giro.mean()),
              "custo_lado": config.CUSTO_LADO}

    for k in sorted({config.SORTEIO_POR_DIA, config.TOP}):
        mac = np.cumprod(1 + _sorteio(ret_por_dia, rng, k), axis=0)
        log = np.log(mac)
        media, desvio = log.mean(1), log.std(1)
        curvas[f"sorteio{k}_media"] = np.exp(media)
        for s in (1, 3):
            curvas[f"sorteio{k}_{s}s_inf"] = np.exp(media - s * desvio)
            curvas[f"sorteio{k}_{s}s_sup"] = np.exp(media + s * desvio)
        pd.DataFrame(mac[:, :200], dias).to_csv(SAIDA / f"sorteio{k}_amostra.csv")
        pd.Series(mac[-1], name="final").to_csv(SAIDA / f"sorteio{k}_finais.csv", index=False)
        resumo[f"sorteio_{k}_por_dia"] = _regua(curvas["modelo_top3"].iloc[-1], mac[-1])
    return curvas, resumo


def backtest_horizonte(pred: pd.DataFrame, ibov: pd.Series, taxa_cdi: pd.Series,
                       semente: int = 42) -> tuple[pd.DataFrame, dict]:
    """Episódio 2: decide a cada HORIZONTE pregões e segura a carteira até a próxima decisão.

    A IA compra as TOP com maior retorno previsto, só se o previsto for positivo; vaga
    sem ação positiva fica no CDI. O sorteio também segura HORIZONTE pregões, para a
    comparação continuar justa. Custo cobrado só no dia em que a carteira troca.
    """
    rng = np.random.default_rng(semente)
    h = config.HORIZONTE
    dias = pd.DatetimeIndex(np.sort(pred["data"].unique()))

    def tabela(c):
        return pred.pivot(index="data", columns="ticker", values=c).reindex(dias)

    ret = np.nan_to_num(tabela("ret_amanha").to_numpy(np.float64))   # dia sem pregão: 0
    pode = tabela("negociavel").fillna(False).to_numpy(bool)
    nota = tabela("score").to_numpy(np.float64)
    cdi_d = taxa_cdi.shift(-1).reindex(dias).ffill().fillna(0).to_numpy()
    n_dias = len(dias)

    modelo, custo = np.zeros(n_dias), np.zeros(n_dias)
    giros, acertos, todos, n_positivos = [], [], [], []
    ontem: set = set()
    sorteios = {k: np.zeros((n_dias, config.SORTEIOS))
                for k in sorted({config.SORTEIO_POR_DIA, config.TOP})}
    for i in range(0, n_dias, h):
        bloco = slice(i, min(i + h, n_dias))
        disp = np.flatnonzero(pode[i] & ~np.isnan(nota[i]))
        if len(disp) == 0:
            modelo[bloco] = cdi_d[bloco]
            continue
        ordem = disp[np.argsort(-nota[i, disp])]
        escolha = [j for j in ordem[:config.TOP] if nota[i, j] > 0]
        n_positivos.append(int((nota[i, disp] > 0).sum()))
        vagas_cdi = config.TOP - len(escolha)
        modelo[bloco] = (ret[bloco][:, escolha].sum(1) + vagas_cdi * cdi_d[bloco]) / config.TOP
        acum = np.prod(1 + ret[bloco], axis=0) - 1
        acertos.extend((acum[escolha] > 0).tolist())
        todos.extend((acum[disp] > 0).tolist())
        carteira = set(escolha)
        giro = len(carteira ^ ontem) / config.TOP
        giros.append(giro)
        custo[i] = giro * config.CUSTO_LADO
        ontem = carteira
        for k, matriz in sorteios.items():
            sorteado = disp[np.argsort(rng.random((config.SORTEIOS, len(disp))), axis=1)[:, :k]]
            for d in range(bloco.start, bloco.stop):
                matriz[d] = ret[d][sorteado].mean(1)

    diario = pd.DataFrame({
        "modelo_top3": modelo,
        "modelo_top3_custo": modelo - custo,
        "ibov": ibov.shift(-1).reindex(dias).fillna(0).to_numpy(),
        "cdi": cdi_d,
    }, index=dias)
    curvas = (1 + diario).cumprod()
    resumo = {"dias": n_dias, "inicio": str(dias[0].date()), "fim": str(dias[-1].date()),
              "horizonte": h, "rebalanceamentos": len(giros),
              "acoes_por_dia": float(pode.sum(1).mean()),
              "curvas": {c: _estatisticas(diario[c]) for c in diario},
              "acerto_top3": float(np.mean(acertos)) if acertos else float("nan"),
              "acerto_uma_qualquer": float(np.mean(todos)),
              "rebalanceamentos_com_positivo": int(np.sum(np.array(n_positivos) > 0)),
              "positivos_medio": float(np.mean(n_positivos)),
              "giro_por_rebalanceamento": float(np.mean(giros)),
              "giro_medio_dia": float(np.sum(giros) / n_dias),
              "custo_lado": config.CUSTO_LADO}
    for k, matriz in sorteios.items():
        mac = np.cumprod(1 + matriz, axis=0)
        log = np.log(mac)
        media, desvio = log.mean(1), log.std(1)
        curvas[f"sorteio{k}_media"] = np.exp(media)
        for s in (1, 3):
            curvas[f"sorteio{k}_{s}s_inf"] = np.exp(media - s * desvio)
            curvas[f"sorteio{k}_{s}s_sup"] = np.exp(media + s * desvio)
        pd.DataFrame(mac[:, :200], dias).to_csv(SAIDA / f"sorteio{k}_amostra.csv")
        pd.Series(mac[-1], name="final").to_csv(SAIDA / f"sorteio{k}_finais.csv", index=False)
        resumo[f"sorteio_{k}_por_dia"] = _regua(curvas["modelo_top3"].iloc[-1], mac[-1])
    return curvas, resumo


def importancia(rede, matriz: np.ndarray, alvo: np.ndarray, fins: np.ndarray,
                colunas: list[str], zero: float, semente: int = 5) -> pd.DataFrame:
    """Importância por permutação: embaralha uma informação de cada vez no teste e mede o
    quanto a perda piora. Se embaralhar e nada mudar, a rede não usava aquilo."""
    rng = np.random.default_rng(semente)
    base, _ = avaliar(rede, matriz, alvo, fins, zero=zero)
    linhas = np.unique((fins[:, None] + np.arange(-config.JANELA + 1, 1)).ravel())
    saida = []
    for j, c in enumerate(colunas):
        m = matriz.copy()
        m[linhas, j] = m[rng.permutation(linhas), j]
        perda, _ = avaliar(rede, m, alvo, fins, zero=zero)
        saida.append({"informacao": c, "aumento_perda": perda - base,
                      "aumento_pct": (perda - base) / base})
    return pd.DataFrame(saida).sort_values("aumento_perda", ascending=False)


# ---------------------------------------------------------------- etapas

def rodar_etapa(n: int, tab, div, epocas_max: int | None, rapido: bool) -> dict:
    nome, extras, epocas, callback = ETAPAS[n]
    if rapido:
        epocas = min(epocas, 1 if n <= 2 else 3)
    if epocas_max:
        epocas = min(epocas, epocas_max)
    colunas = FEATURES + extras
    print(f"\n{n}. {nome}: {len(colunas)} features, {epocas} época(s), "
          f"callback {'ligado' if callback else 'desligado'}")

    norm = normalizacao(tab, colunas, div["corte_validacao"])
    matriz = escalar(tab, norm)
    alvo = tab[RESPOSTA].to_numpy(np.float32)
    bruto = alvo
    zero = 0.0
    if config.ALVO == "retorno":
        # O retorno é padronizado com média e desvio só do treino, como as features.
        mu = float(np.nanmean(alvo[div["treino"]]))
        sd = float(np.nanstd(alvo[div["treino"]]))
        alvo = ((alvo - mu) / sd).astype(np.float32)
        zero = -mu / sd
        norm["alvo"] = {"media": mu, "desvio": sd}
    pesos = None
    if config.MEIA_VIDA:
        # O dado recente pesa mais: a cada MEIA_VIDA anos para trás, o peso cai pela metade.
        idade = (div["corte_validacao"] - tab["data"]).dt.days.clip(lower=0) / 365.25
        pesos = (0.5 ** (idade / config.MEIA_VIDA)).to_numpy(np.float32)
    rede, log = treinar(matriz, alvo, div["treino"], div["validacao"], epocas,
                        callback=callback, rotulo=nome, pesos=pesos, zero=zero)
    pd.DataFrame(log).to_csv(SAIDA / f"log_{nome}.csv", index=False)
    _, acerto = avaliar(rede, matriz, alvo, div["teste"], zero=zero)
    print(f"  acerto no teste: {acerto:.1%}  (sempre dizer 'sobe' dá "
          f"{(bruto[div['teste']] > 0).mean():.1%})")

    resultado = {"etapa": n, "nome": nome, "extras": extras, "epocas_pedidas": epocas,
                 "epocas_rodadas": len(log), "callback": callback,
                 "acerto_treino_final": log[-1]["acerto_treino"],
                 "acerto_validacao_final": log[-1]["acerto_validacao"],
                 "acerto_validacao_melhor": max(e["acerto_validacao"] for e in log),
                 "acerto_teste": acerto}

    # O backtest roda na etapa 1 (o "olha que beleza") e na final do episódio.
    if n in (1, config.FINAL):
        print(f"  prevendo o teste com dropout de Monte Carlo ({config.PASSADAS_MC} passadas)...")
        prob, incerteza = prever(rede, matriz, div["teste"])
        pred = tab.iloc[div["teste"]][["data", "ticker", "ret_amanha", RESPOSTA, "negociavel"]].copy()
        if config.ALVO == "retorno":
            a = norm["alvo"]
            pred["score"] = prob * a["desvio"] + a["media"]     # retorno em log previsto
            pred["incerteza"] = incerteza * a["desvio"]
            pred["sinal"] = np.where(pred["score"] > 0, "comprar", "ficar no CDI")
        else:
            pred["prob"], pred["incerteza"] = prob, incerteza
            pred["sinal"] = np.where(prob >= config.LIMIAR, "comprar", "manter")
        pred.to_csv(SAIDA / f"previsoes_{nome}.csv", index=False)
        ibov = tab.drop_duplicates("data").set_index("data")["ibov"].sort_index()
        if config.ALVO == "sobe" and config.HORIZONTE == 1:
            curvas, bt = backtest(pred, ibov, cdi())                # o do episódio 1, intocado
        else:
            curvas, bt = backtest_horizonte(pred, ibov, cdi())
        curvas.to_csv(SAIDA / f"curvas_{nome}.csv")
        if config.ALVO == "retorno":
            resultado["previsto"] = {"min": float(pred["score"].min()),
                                     "max": float(pred["score"].max()),
                                     "positivos": float((pred["score"] > 0).mean()),
                                     "incerteza_media": float(pred["incerteza"].mean())}
        else:
            resultado["prob"] = {"min": float(prob.min()), "max": float(prob.max()),
                                 "media": float(prob.mean()),
                                 "acima_limiar": int((prob >= config.LIMIAR).sum()),
                                 "incerteza_media": float(incerteza.mean())}
        resultado["backtest"] = bt
        c, s = bt["curvas"], bt[f"sorteio_{config.SORTEIO_POR_DIA}_por_dia"]
        print(f"  top 3 {c['modelo_top3']['multiplicador']:.2f}x "
              f"(com custo {c['modelo_top3_custo']['multiplicador']:.2f}x)   "
              f"Ibovespa {c['ibov']['multiplicador']:.2f}x   CDI {c['cdi']['multiplicador']:.2f}x   "
              f"sorteio {s['media']:.2f}x   z do modelo {s['z_modelo']:+.2f}")

    if n == config.FINAL:
        torch.save(rede.state_dict(), SAIDA / "modelo_final.pt")
        (SAIDA / "normalizacao.json").write_text(json.dumps(norm), encoding="utf-8")
        if config.EPISODIO >= 2:
            print("  importância por permutação (uma informação embaralhada por vez)...")
            imp = importancia(rede, matriz, alvo, div["teste"], colunas, zero)
            imp.to_csv(SAIDA / "importancia.csv", index=False)
            resultado["importancia_top5"] = imp.head(5).to_dict("records")
            print(imp.head(8).to_string(index=False))
    return resultado


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodio", type=int, default=config.ATUAL)  # lido pelo config.py
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--etapas", default="1,2,3,4,5")
    ap.add_argument("--epocas-max", type=int, default=None)
    args = ap.parse_args()
    etapas = [int(e) for e in args.etapas.split(",")]
    SAIDA.mkdir(parents=True, exist_ok=True)
    relogio = time.time()

    print("montando as features...")
    tab = montar()
    fins = fins_de_janela(tab)
    div = dividir(tab, fins)
    teste = div["teste"]
    print(f"  {tab['ticker'].nunique()} ações, {len(fins):,} janelas de {config.JANELA} dias")
    print(f"  treino até {div['corte_validacao'].date()}, validação até "
          f"{div['corte_teste'].date()}, teste depois")

    arquivo = SAIDA / "resumo.json"
    resumo = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}
    resumo.update({
        "acoes": sorted(tab["ticker"].unique()), "janelas": int(len(fins)),
        "features": FEATURES, "horizonte": config.HORIZONTE, "janela": config.JANELA,
        "periodo": [str(tab["data"].min().date()), str(tab["data"].max().date())],
        "corte_validacao": str(div["corte_validacao"].date()),
        "corte_teste": str(div["corte_teste"].date()),
        "n_treino": int(len(div["treino"])), "n_validacao": int(len(div["validacao"])),
        "n_teste": int(len(teste)),
        "episodio": config.EPISODIO, "alvo": config.ALVO, "meia_vida": config.MEIA_VIDA,
        "sempre_sobe_teste": float((tab[RESPOSTA].to_numpy()[teste] > 0).mean()),
        "insider_coberto_treino": float(tab["insider_coberto"].to_numpy()[div["treino"]].mean()),
        "dias_com_noticia": float((tab["n_noticias"] > 0).mean()),
        "dias_com_insider_comprando": float(tab["insider_comprando"].mean()),
    })
    resumo.setdefault("etapas", {})

    for n in etapas:
        resumo["etapas"][str(n)] = rodar_etapa(n, tab, div, args.epocas_max, args.rapido)
        resumo["minutos"] = round((time.time() - relogio) / 60, 1)
        arquivo.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\npronto em {resumo['minutos']} min. Previsão: python prever.py --episodio {config.EPISODIO}")


if __name__ == "__main__":
    main()
