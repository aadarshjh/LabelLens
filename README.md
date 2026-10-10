# LabelLens: AI-Powered Food Ingredient Recognition and Analysis

LabelLens is an undergraduate AI project that reads packaged-food labels, identifies ingredients, normalizes extracted names, and retrieves curated evidence with source provenance. It supports consumer explanations; it does not certify food safety or provide medical advice.

## Architecture

```text
Food Label Image -> OCR -> Extraction -> Normalization -> RAG -> Groq LLM -> Validation -> Frontend
```

The system performs OCR, ingredient extraction, conservative normalization, evidence retrieval via RAG, constrained Groq explanation via a single batch request, validation against pipeline invariants, and consumer presentation. RAG retrieves relevant evidence from the curated knowledge base before the LLM generates an explanation. RAG grounds the explanation on retrieved text but does not guarantee hallucination prevention.

## Technologies

- **Python**: Core pipeline orchestration and runtime
- **OpenCV**: Image preprocessing strategies for food labels
- **Tesseract**: Optical character recognition (PSM selection & confidence scoring)
- **RapidFuzz**: Conservative fuzzy matching against ingredient dictionary
- **Sentence Transformers** (`all-MiniLM-L6-v2`): Semantic embeddings for knowledge retrieval
- **FAISS**: Vector indexing and fast similarity retrieval
- **Groq**: High-throughput LLM explanation using structured JSON output
- **HTML/CSS/JavaScript**: Clean, consumer-facing user interface
- **pytest**: Automated unit and integration test suite

## Project Layout

```text
src/          OCR, extraction, normalization, RAG, LLM, pipeline, frontend adapters
data/         ingredient dictionary and knowledge-base documents
vectorstore/  FAISS index and provenance metadata
tests/        unit and integration tests
evaluation/   metrics, grounding checks, and recorded results
frontend/     static consumer interface
docs/         architecture and setup notes
artifacts/    local reference and runtime outputs (ignored)
```

## Setup and Demo

Use Python 3.12 or newer (the pinned NumPy release requires it). Install the system Tesseract OCR executable as well as the Python packages below; `pytesseract` is only the Python interface. On macOS, install it with `brew install tesseract`; on Debian or Ubuntu, use `sudo apt-get install tesseract-ocr`. Ensure `tesseract` is on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf download sentence-transformers/all-MiniLM-L6-v2
python run_demo.py path/to/label-photo.jpeg
```

Supply your own label image to the demo. The sample photos used during development are not included because redistribution permission has not been confirmed.

The `hf download` command caches the embedding model used by retrieval. Runtime retrieval loads it from the local cache and does not download it implicitly. `requirements.txt` pins the direct Python dependencies used by the verified environment. The Docker image downloads the pinned `all-MiniLM-L6-v2` snapshot at build time and installs Tesseract, so neither depends on a host's local cache.

The default demo runs deterministically through the pre-LLM boundary without calling an external provider. To attempt live Groq analysis, set `GROQ_API_KEY` in your environment and use:

```bash
python run_demo.py --with-llm
```

Run the local web interface with (the server binds to `0.0.0.0`; `PORT` defaults to `8000`):

```bash
python run_frontend.py
```
and open `http://localhost:8000`.

### Docker

Build and run the production-style container from the repository root:

```bash
docker build -t labellens .
docker run --rm -p 8000:8000 labellens
```

The host port may differ; set the container's `PORT` to the port exposed by the hosting platform. `GROQ_API_KEY` is read only by the backend. `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`. For local live Groq analysis, create `.env` from `.env.example` and run with `--env-file .env`; the `.dockerignore` keeps it out of the build context. Do not pass a key as a Docker build argument. The `.dockerignore` also excludes Git data, local images, caches, logs, and generated reports.

### Render deployment

After publishing the approved repository to GitHub, create a Render **Web Service** from that repository and select **Docker** as the runtime. Render builds the root `Dockerfile`; set `GROQ_API_KEY` in the service's Environment settings (and optionally `GROQ_MODEL`). Render supplies `PORT`; the app binds to `0.0.0.0` and uses that value. Deploy, then open `/scan.html` at the assigned `onrender.com` URL and upload each demo image. Confirm a successful health-free page load and actual scan response before describing the service as deployed. This repository has not been deployed or tested at a public URL.

The local demo images are `label1.jpeg`, `label2.jpeg`, and `label3.jpeg` in the project root. They are excluded from the Docker build context and remain under redistribution-rights review for the public repository. Provide them separately for the three-label demonstration unless publication rights are confirmed. Label 2 is a 327 × 152 pixel beverage panel with small, curved, blurred print. Upscaling recovers some ingredient codes, but several line boundaries and words remain unreadable; these are displayed as unclear/review items. Treat its extraction as partial and check the source image before relying on any ingredient identity. Do not interpret a successful scan as proof of OCR accuracy.

For the browser demonstration, run `python run_frontend.py`, open `http://127.0.0.1:8000/scan.html`, and upload a label photo supplied separately. The upload calls the local backend and renders its results page.

For live Groq analysis, copy `.env.example` to `.env` and set `GROQ_API_KEY`. `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`. Keep `.env` private; it is excluded by `.gitignore`.

Run the public, deterministic regression suite after installing `requirements.txt`:

```bash
PYTHONPATH="$(pwd)" python -m pytest -q tests
# Or use the same public default through the project runner:
PYTHONPATH="$(pwd)" bash run_all_tests.sh
```

`pytest.ini` registers the `model` and `private_data` markers and excludes those prerequisite-dependent tests from the default public suite. To run the semantic retrieval checks, first download the documented `all-MiniLM-L6-v2` model, then run:

```bash
PYTHONPATH="$(pwd)" python -m pytest -q tests -m "model and not private_data"
```

The private-data RAG evaluation additionally requires local `data/ground_truth/ground_truth.json` and `artifacts/reference_runs/extraction/extraction_results.json` files, which are not distributed. With those files and the model available, run:

```bash
PYTHONPATH="$(pwd)" python -m pytest -q tests -m private_data
```

The synthetic grounding tests use only public test support and do not call live LLM services. Evaluation reports are generated locally under the ignored `evaluation/results/` directory.

OCR evaluates multiple preprocessing strategies and Tesseract segmentation modes. Extraction detects the ingredient section and parses compound ingredients. Normalization prioritizes exact names, aliases, and codes, then uses conservative fuzzy matching; uncertain identities remain `ambiguous` or `unknown`. RAG uses exact lookup and thresholded FAISS retrieval, preserving source provenance.

Run `python -m pytest tests` after installing root requirements. Live Groq analysis remains dependent on API key availability, network connectivity, and quota; the repository does not claim live LLM success without a valid response.
