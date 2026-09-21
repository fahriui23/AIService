import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import STORAGE_DIR, REPORTS_DIR, EVALUATION_MODE
from app.database.db import init_db
from app.api.endpoints import router as api_router

service_name = os.environ.get("ABSA_SERVICE_NAME", "ABSA Multi-Engine Web API")
service_version = os.environ.get("ABSA_SERVICE_VERSION", "15.1.0")
engine_only = os.environ.get("ABSA_ENGINE_ONLY", "").strip().lower()

app = FastAPI(
    title=service_name,
    description=(
        "Dedicated V14 aspect, opinion, and sentiment inference API. "
        "Relation and taxonomy remain review-only fallbacks."
        if engine_only == "v14"
        else "End-to-End ABSA Pipeline API supporting V11, V12, V14, and V15.1 multi-aspect inference"
    ),
    version=service_version,
)

# CORS setup
origins = [origin.strip() for origin in os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

# Mount API router
app.include_router(api_router, prefix="/api")

# Static files for report downloads
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
