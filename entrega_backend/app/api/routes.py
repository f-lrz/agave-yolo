from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.vision_service import process_image

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Recebe UMA foto de drone e devolve a imagem marcada + o laudo em JSON.

    Contrato completo (campos, tempos, avisos) em API.md. Os erros 400 trazem
    mensagens em português, prontas para o front exibir.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"O arquivo {file.filename} não é uma imagem válida")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"O arquivo {file.filename} está vazio")

        # Detecção + triagem de saúde. Leva ~12s e roda fora do event loop
        # (ver process_image), então o servidor segue respondendo enquanto isso.
        resultado = await process_image(contents)

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            # mantido para não quebrar o frontend, que lê data.image_data
            "image_data": f"data:image/jpeg;base64,{resultado['image_base64']}",
            "total_agaves": resultado["total_agaves"],
            "largura": resultado["largura"],
            "altura": resultado["altura"],
            "conf": resultado["conf"],
            # Se vier preenchido, MOSTRE. Casos: imagem redimensionada e foto
            # tirada baixo demais — os dois derrubam o resultado sem dar erro.
            "avisos": resultado["avisos"],
            # Triagem de saúde (etapa 2). `plantas` vem ordenada da MAIS
            # suspeita para a menos: é a fila de revisão do agrônomo.
            "saude_ativa": resultado["saude_ativa"],
            "limiar_doente": resultado["limiar_doente"],
            "doentes": resultado["doentes"],
            "sadias": resultado["sadias"],
            "taxa_doenca": resultado["taxa_doenca"],
            "plantas": resultado["plantas"],
        }

    except HTTPException:
        raise
    except ValueError as e:
        # imagem corrompida / não decodificável é erro do cliente, não do servidor
        raise HTTPException(status_code=400, detail=f"Erro ao ler a imagem {file.filename}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem {file.filename}: {str(e)}")
