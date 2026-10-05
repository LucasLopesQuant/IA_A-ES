# -*- coding: utf-8 -*-
"""
Baixa ou atualiza tudo: preço e CDI, o mapa ticker -> CNPJ, insider e notícias.

    python baixar.py              # tudo
    python baixar.py precos       # só uma fonte (precos, mapa, insiders, noticias)

Rodar de novo só pega o que é novo. Nada aqui treina modelo.
"""
from __future__ import annotations

import sys

import cvm
import insiders
import noticias
import precos

FONTES = {"precos": precos.atualizar, "mapa": cvm.atualizar_mapa,
          "insiders": insiders.atualizar, "noticias": noticias.atualizar}


def atualizar_tudo(fontes: list[str] | None = None) -> None:
    for nome in fontes or FONTES:
        print(f"\n{nome}")
        FONTES[nome]()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    atualizar_tudo(sys.argv[1:] or None)
