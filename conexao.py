# -*- coding: utf-8 -*-
"""
Rede e chaves, num lugar só.

IPv4 forçado: em rede com IPv6 sem rota, o `requests` fica cerca de 168 s pendurado
em cada chamada antes de cair no IPv4. Se a sua rede não tem esse problema, pode
desligar com FORCE_IPV4 = False.
"""
from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import requests

FORCE_IPV4 = True

if FORCE_IPV4:
    _getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = lambda *a, **kw: [
        r for r in _getaddrinfo(*a, **kw) if r[0] == socket.AF_INET
    ]

AQUI = Path(__file__).resolve().parent
NAVEGADOR = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def ler_env(nome: str) -> str | None:
    """Variável de ambiente primeiro, depois um .env nesta pasta ou nas duas de cima."""
    valor = os.getenv(nome)
    if valor:
        return valor.strip()
    for pasta in (AQUI, *AQUI.parents[:2]):
        arquivo = pasta / ".env"
        if not arquivo.exists():
            continue
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            chave, _, resto = linha.partition("=")
            if chave.strip() == nome and resto.strip():
                return resto.strip().strip('"').strip("'")
    return None


def baixar(url: str, params: dict | None = None, agente: str = NAVEGADOR,
           tentativas: int = 3, **kw) -> requests.Response:
    """GET com retentativa. Erro 4xx volta na hora, porque repetir não resolve."""
    for i in range(tentativas):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": agente},
                             timeout=60, **kw)
        except requests.RequestException:
            if i == tentativas - 1:
                raise
            time.sleep(2 * (i + 1))
            continue
        if r.status_code < 500 and r.status_code != 429:
            return r
        time.sleep(2 * (i + 1))
    return r
