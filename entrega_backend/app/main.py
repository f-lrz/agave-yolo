from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import setup_cors
from app.api.routes import router as api_router
from app.services.vision_service import USAR_MOCK, obter_modelos


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega os DOIS modelos (detector + saúde) ANTES de aceitar requisição.
    # Sem isto, o primeiro usuário paga ~5s extras (import do ultralytics +
    # leitura dos pesos) além dos ~12s da inferência.
    if not USAR_MOCK:
        obter_modelos()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="IDENTIFY 2.0 Backend",
    # Etapa 1: detecta as agaves. Etapa 2: triagem de saúde — cada planta
    # recebe uma probabilidade de doença para o agrônomo conferir em campo.
    # É triagem, não diagnóstico (ver models/MODELO_SAUDE.md).
    description="API de Visão Computacional para detecção de plantas de agave "
                "em imagens de drone e triagem de suspeita de doença. A triagem "
                "aponta plantas para conferência em campo; não é diagnóstico.",
    version="2.0.0"
)

setup_cors(app)
app.include_router(api_router, prefix="/api")

# Serve a pasta static/
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rota raiz serve a página de upload/loading/resultado
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Health check - /health
@app.get("/health")
async def health():
    return {"status": "online", "message": "Backend do IDENTIFY 2.0 ativo."}
