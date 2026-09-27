# MLOPS Project - DOCKER
#
# One image, three roles (selected by the `command` in docker-compose.yml):
#   - api     -> FastAPI inference service + web UI  (port 3000)
#   - mlflow  -> MLflow tracking server + registry   (port 5001)
#   - jupyter -> JupyterLab for training/experiments (port 8888)
# Using a single image keeps library versions identical between the place a
# model is trained (Jupyter), where it is tracked (MLflow) and where it is served (API).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/Models \
    PYTHONPATH=/app/src

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de requisitos primeiro (isso acelera futuros builds)
COPY docker/requirements.txt .

# Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# Copia as pastas essenciais do projeto para dentro da imagem
# (Models is baked in so the API image works standalone; compose mounts the
#  live folder on top so models retrained in Jupyter are picked up.)
COPY api /app/api
COPY src /app/src
COPY Models /app/Models

# Portas: API, MLflow, Jupyter
EXPOSE 3000 5001 8888

# Default role = API. python is used for the probe because curl is not in the slim image.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "3000"]
