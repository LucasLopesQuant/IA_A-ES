# -*- coding: utf-8 -*-
"""
A rede do vídeo: convolução 1D, depois LSTM, com dropout de Monte Carlo.

A convolução olha três dias por vez e acha picos, vales e tendência curta. A LSTM
lê a janela inteira em ordem. Na hora de prever o dropout continua ligado, e a
mesma janela passa PASSADAS_MC vezes: a média é a previsão, o desvio é a incerteza.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

import config
from features import janelas


class Rede(nn.Module):
    def __init__(self, n_features: int, canais: int = 32, oculto: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(n_features, canais, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(canais, oculto, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.densa = nn.Linear(oculto, 16)
        self.saida = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (lote, dias, features)
        h = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (ultimo, _) = self.lstm(self.drop(h))
        h = torch.relu(self.densa(self.drop(ultimo[-1])))
        return self.saida(self.drop(h)).squeeze(-1)


def _lotes(fins: np.ndarray, tamanho: int, rng: np.random.Generator | None = None):
    ordem = rng.permutation(len(fins)) if rng is not None else np.arange(len(fins))
    for i in range(0, len(fins), tamanho):
        yield fins[ordem[i:i + tamanho]]


REGRESSAO = config.ALVO == "retorno"


def _perda(reducao: str = "mean") -> nn.Module:
    """Sobe ou não (episódio 1): entropia cruzada. Quanto (episódio 2): erro quadrático."""
    return nn.MSELoss(reduction=reducao) if REGRESSAO else nn.BCEWithLogitsLoss(reduction=reducao)


def avaliar(rede: Rede, matriz: np.ndarray, alvo: np.ndarray, fins: np.ndarray,
            lote: int = 8192, zero: float = 0.0) -> tuple[float, float]:
    """Perda e acerto com o dropout desligado.

    Acerto é acertar o sinal: na classificação, a probabilidade passar de 50%; na
    regressão, prever subida quando subiu. `zero` é onde fica o retorno zero na escala
    em que a rede treina (o alvo da regressão é padronizado).
    """
    rede.eval()
    perda, acertos = 0.0, 0
    fn = _perda("sum")
    with torch.no_grad():
        for idx in _lotes(fins, lote):
            z = rede(torch.from_numpy(janelas(matriz, idx)))
            y = torch.from_numpy(alvo[idx])
            perda += fn(z, y).item()
            if REGRESSAO:
                acertos += ((z > zero) == (y > zero)).sum().item()
            else:
                acertos += ((z > 0).float() == y).sum().item()
    return perda / len(fins), acertos / len(fins)


def treinar(matriz: np.ndarray, alvo: np.ndarray, treino: np.ndarray, validacao: np.ndarray,
            epocas: int, callback: bool = True, paciencia: int = 10,
            rotulo: str = "", pesos: np.ndarray | None = None,
            zero: float = 0.0) -> tuple[Rede, list[dict]]:
    """Treina e devolve a rede e o log por época.

    Com `callback`, é o EarlyStopping do Keras: para quando a perda na validação passa
    `paciencia` épocas sem melhorar, e volta para os pesos da melhor época. Sem ele,
    roda todas as épocas e fica com os pesos da última, como no fim do vídeo.

    `pesos` (episódio 2) dá a cada janela um peso no treino, para o dado recente contar
    mais. Sem pesos, é o treino do episódio 1, idêntico.
    """
    torch.manual_seed(config.SEMENTE)
    rng = np.random.default_rng(config.SEMENTE)
    rede = Rede(matriz.shape[1])
    otimizador = torch.optim.Adam(rede.parameters(), lr=config.TAXA)
    fn = _perda()
    fn_cada = _perda("none")
    amostra_treino = treino[:: max(1, len(treino) // len(validacao))] if len(validacao) else treino

    log, melhor, estado, sem_melhora = [], np.inf, None, 0
    for epoca in range(1, epocas + 1):
        t0 = time.time()
        rede.train()
        # Ctrl+C encerra o treino aqui, com os pesos da última época inteira, e o
        # experimento segue para a previsão e o backtest em vez de perder tudo.
        copia = {k: v.clone() for k, v in rede.state_dict().items()}
        try:
            for idx in _lotes(treino, config.LOTE, rng):
                otimizador.zero_grad()
                z = rede(torch.from_numpy(janelas(matriz, idx)))
                y = torch.from_numpy(alvo[idx])
                if pesos is None:
                    fn(z, y).backward()
                else:
                    w = torch.from_numpy(pesos[idx])
                    ((fn_cada(z, y) * w).sum() / w.sum()).backward()
                otimizador.step()
        except KeyboardInterrupt:
            rede.load_state_dict(copia)
            print(f"\n  {rotulo} interrompido na época {epoca}: segue com {len(log)} épocas")
            break

        p_tr, a_tr = avaliar(rede, matriz, alvo, amostra_treino, zero=zero)
        p_va, a_va = avaliar(rede, matriz, alvo, validacao, zero=zero)
        log.append({"epoca": epoca, "perda_treino": p_tr, "acerto_treino": a_tr,
                    "perda_validacao": p_va, "acerto_validacao": a_va})
        print(f"  {rotulo} época {epoca:>3}/{epocas}  treino {a_tr:.1%}  validação {a_va:.1%}"
              f"  perda val {p_va:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        if not callback:
            continue
        if p_va < melhor - 1e-4:
            melhor, sem_melhora = p_va, 0
            estado = {k: v.clone() for k, v in rede.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"  {rotulo} callback parou o treino na época {epoca}")
                break

    if callback and estado is not None:
        rede.load_state_dict(estado)
    return rede, log


def prever(rede: Rede, matriz: np.ndarray, fins: np.ndarray, passadas: int = config.PASSADAS_MC,
           lote: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidade de subir (ou o retorno previsto, na regressão) e a incerteza, com o
    dropout ligado de propósito."""
    rede.train()
    medias, desvios = [], []
    with torch.no_grad():
        for idx in _lotes(fins, lote):
            x = torch.from_numpy(janelas(matriz, idx))
            saida = (lambda t: t) if REGRESSAO else torch.sigmoid
            p = torch.stack([saida(rede(x)) for _ in range(passadas)])
            medias.append(p.mean(0).numpy())
            desvios.append(p.std(0, correction=0).numpy())
    rede.eval()
    return np.concatenate(medias), np.concatenate(desvios)
