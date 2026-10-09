FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Pin the model snapshot so the image does not depend on a developer's cache
# or a moving upstream `main` revision.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"

COPY src ./src
COPY frontend ./frontend
COPY data/ingredients.json ./data/ingredients.json
COPY data/knowledge_base ./data/knowledge_base
COPY vectorstore ./vectorstore
COPY run_frontend.py ./run_frontend.py

EXPOSE 8000
CMD ["python", "run_frontend.py"]
