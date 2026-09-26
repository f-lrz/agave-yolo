"""Exemplo de integracao no backend FastAPI existente.

Nao e para rodar como esta — e o padrao a copiar para `app/services/` e
`app/api/`. Mostra as tres coisas que a aplicacao precisa: detectar varias
fotos, guardar o resultado, e servir o download da imagem marcada.

O que este arquivo demonstra e que NAO e obvio:

  1. o modelo carrega UMA vez, num singleton — nao por requisicao;
  2. varias fotos sao processadas em SEQUENCIA, nunca em paralelo;
  3. o download redesenha a partir das caixas guardadas, sem rodar a
     deteccao de novo.
"""

from __future__ import annotations

import io
import uuid
from typing import Any

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from agave_detector import CONF_PADRAO, DetectorAgave, desenhar_caixas

# --------------------------------------------------------------------------
# app/services/agave.py
# --------------------------------------------------------------------------

_detector: DetectorAgave | None = None


def obter_detector() -> DetectorAgave:
    """Singleton. Carregar o modelo por requisicao multiplicaria a latencia.

    Prefira instanciar no startup do app (evento `lifespan`) para que a
    primeira requisicao nao pague o carregamento.
    """
    global _detector
    if _detector is None:
        _detector = DetectorAgave("modelo/best.pt")   # AGAVE_PESOS, se preferir env
    return _detector


# Substitua por banco/storage de verdade. Guardar em memoria some no restart e
# nao funciona com mais de uma replica.
ARMAZEM: dict[str, dict[str, Any]] = {}


def processar(nome: str, dados: bytes, conf: float = CONF_PADRAO) -> dict[str, Any]:
    """Detecta em UMA foto e guarda o necessario para o download depois.

    Guardamos os bytes originais + as caixas. Guardar a imagem ja desenhada
    custaria ~6,6 MB por foto e travaria a cor/espessura das caixas; redesenhar
    e rapido, porque nao envolve inferencia.
    """
    det = obter_detector()
    r = det.detectar(dados, conf=conf)
    ident = uuid.uuid4().hex
    ARMAZEM[ident] = {"nome": nome, "bytes": dados, "caixas": r["caixas"]}
    return {"id": ident, "nome": nome, "total": r["total"],
            "largura": r["largura"], "altura": r["altura"],
            "avisos": r["avisos"]}


# --------------------------------------------------------------------------
# app/api/agave.py
# --------------------------------------------------------------------------

router = APIRouter(prefix="/agave", tags=["agave"])

MAX_FOTOS = 20
SEGUNDOS_POR_FOTO = 11        # medido em CPU de 12 nucleos, 5472x3078


@router.post("/detectar")
async def detectar(arquivos: list[UploadFile] = File(...)) -> dict:
    """Aceita varias fotos. Devolve um resumo por foto, com o id do download.

    ATENCAO: ~11s por foto, em sequencia. Com 10 fotos sao ~2 minutos, o que
    estoura o timeout de qualquer proxy razoavel. Em producao, troque este
    endpoint por: enfileirar -> 202 com o id do job -> o front consulta o
    progresso. O laco abaixo continua valendo, mudando so quem o executa.
    """
    if not arquivos:
        raise HTTPException(400, "nenhum arquivo enviado")
    if len(arquivos) > MAX_FOTOS:
        raise HTTPException(413, f"maximo de {MAX_FOTOS} fotos por requisicao "
                                 f"(~{MAX_FOTOS * SEGUNDOS_POR_FOTO}s)")

    resultados, erros = [], []
    for arq in arquivos:
        dados = await arq.read()
        if not dados:
            erros.append({"nome": arq.filename, "erro": "arquivo vazio"})
            continue
        try:
            # Em sequencia, de proposito. Cada foto de 5472x3078 ocupa ~50 MB
            # decodificada, e a inferencia ja usa todos os nucleos: processar
            # em paralelo so multiplicaria a RAM sem ganhar tempo.
            resultados.append(processar(arq.filename or "sem_nome", dados))
        except ValueError as e:
            erros.append({"nome": arq.filename, "erro": str(e)})

    return {"fotos": resultados, "erros": erros,
            "total_agaves": sum(r["total"] for r in resultados)}


@router.get("/{ident}/imagem")
def baixar_imagem(ident: str) -> StreamingResponse:
    """Download da foto com as caixas desenhadas.

    NAO roda a deteccao de novo: redesenha a partir das caixas guardadas. Sao
    milissegundos em vez dos 11 segundos da inferencia.
    """
    item = ARMAZEM.get(ident)
    if item is None:
        raise HTTPException(404, "id nao encontrado")

    img = desenhar_caixas(item["bytes"], item["caixas"])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "falha ao codificar a imagem")

    nome = item["nome"].rsplit(".", 1)[0] + "_detectado.jpg"
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        # `attachment` e o que faz o navegador baixar em vez de exibir — e o
        # que o botao de download precisa.
        headers={"Content-Disposition": f'attachment; filename="{nome}"',
                 "X-Total-Agaves": str(len(item["caixas"]))},
    )
