# MLOPS Project - DOCKER

FROM python:3.13-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de requisitos primeiro (isso acelera futuros builds)
COPY docker/requirements.txt .

# Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# Copia as pastas essenciais do projeto para dentro da imagem
COPY api /app/api
COPY src /app/src

#Porta
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["python", "api/app.py"]