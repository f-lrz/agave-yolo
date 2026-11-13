"""Confere a instalacao da entrega no backend. Rode da RAIZ do identify2.0-backend:

    python verificar_entrega.py caminho/para/uma_foto_do_drone.JPG

Checa os pesos (existem? sao os certos?), carrega os dois modelos pelo mesmo
caminho que a API usa (app.services.vision_service), analisa a foto e salva a
imagem desenhada em verificacao_saida.jpg. Nao sobe servidor e nao precisa de
nenhuma dependencia alem do requirements.txt.

Pode apagar este arquivo depois de conferir.
"""

from __future__ import annotations

import base64
import hashlib
import sys
import time
from pathlib import Path

ESPERADO = {
    "models/best.pt":
        "a18006d57162cb8d3e163751321fb3d322ca47ec406a419906e254857292e96e",
    "models/classificador_saude.pt":
        "da9834ce1f434ebf6de0d5aef0187f7de11d1b1de0fecabbda95afe9960964c4",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    foto = Path(sys.argv[1])
    if not foto.exists():
        raise SystemExit(f"foto nao encontrada: {foto}")

    print("== pesos")
    ok = True
    for caminho, sha in ESPERADO.items():
        p = Path(caminho)
        if not p.exists():
            print(f"  FALTA  {caminho}  (rode da raiz do backend? copiou models/?)")
            ok = False
            continue
        real = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"  {'ok    ' if real == sha else 'DIFERE'} {caminho}")
        ok &= real == sha
    if not ok:
        raise SystemExit("\npesos faltando ou diferentes do entregue — confira a copia.")

    print("\n== carregando os modelos (como no startup da API)")
    sys.path.insert(0, str(Path.cwd()))
    from app.services import vision_service as vs
    t0 = time.perf_counter()
    vs.obter_modelos()
    print(f"  ok em {time.perf_counter() - t0:.1f}s  (saude ativa: {vs.USAR_SAUDE}, "
          f"device: {vs.DEVICE}, limiar: {vs.LIMIAR_DOENTE})")

    print(f"\n== analisando {foto.name}")
    t0 = time.perf_counter()
    r = vs._analisar_sincrono(foto.read_bytes())
    print(f"  {time.perf_counter() - t0:.1f}s | {r['total_agaves']} agaves | "
          f"{r['doentes']} suspeitas de doenca | taxa {r['taxa_doenca']}")
    for a in r["avisos"]:
        print(f"  AVISO: {a}")
    print("  5 mais suspeitas (prob_doente):",
          ", ".join(f"{p['prob_doente']:.2f}" for p in r["plantas"][:5]))

    saida = Path("verificacao_saida.jpg")
    saida.write_bytes(base64.b64decode(r["image_base64"]))
    print(f"\n== tudo certo. Imagem desenhada em {saida.resolve()}")


if __name__ == "__main__":
    main()
