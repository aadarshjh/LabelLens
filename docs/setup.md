# Setup

Use Python 3.10 or newer. In the project root, create and activate a virtual environment, then install the pinned Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the Tesseract OCR executable separately (`brew install tesseract` on macOS or `sudo apt-get install tesseract-ocr` on Debian/Ubuntu) and ensure `tesseract` is on `PATH`; `pytesseract` alone is not the OCR engine. Before running the app, cache the retrieval model with `hf download sentence-transformers/all-MiniLM-L6-v2`. Retrieval uses local files only at runtime. The FAISS index and its metadata are checked into the project and can be rebuilt from the knowledge base with `python -m src.rag.main --build`.

Use `python run_demo.py [path/to/image]` for the deterministic pre-LLM demo. The three demonstration images are `label1.jpeg`, `label2.jpeg`, and `label3.jpeg` in the project root. Label 2 is a 327 × 152 pixel curved, blurred beverage panel; its scan returns partial extraction with unclear items. Check the displayed results against the image. For a live Groq attempt, copy `.env.example` to `.env` and set `GROQ_API_KEY`; `GROQ_MODEL` is optional and defaults to `openai/gpt-oss-120b`. `.env` files are ignored and must not be committed. Start the browser app with `python run_frontend.py` and open `http://localhost:8000`.

Run the regression suite with `PYTHONPATH=$(pwd) bash run_all_tests.sh`.
