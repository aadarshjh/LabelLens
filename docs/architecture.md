# LabelLens Architecture

```text
Image -> OCR -> Extraction -> Normalization -> Knowledge Base / RAG
      -> LLM -> Validation -> Consumer UI
```

| Stage | Input | Processing | Output | Purpose |
| --- | --- | --- | --- | --- |
| OCR | Image | Preprocessing and Tesseract candidate selection | text and confidence | What text is present on the label? |
| Extraction | OCR text | Section detection and bracket-aware parsing | ingredient tree | What ingredients are present? |
| Normalization | Parsed ingredient | exact/alias/code/fuzzy matching | canonical name and status | What canonical identities can be assigned? |
| RAG | Normalized ingredient | exact lookup and thresholded FAISS retrieval | evidence and provenance | What evidence do we have? |
| LLM | status and evidence | Gemini structured explanation | candidate result | How do we explain retrieved evidence? |
| Validation | candidate result | schema, status, no-evidence checks | safe result | Does the result remain consistent with deterministic stages? |
| Consumer UI | safe result | serializer and static frontend | browser payload | Present results to the user |
| Evaluation | recorded inputs and outputs | OCR, extraction, normalization, RAG, and grounding metrics | reports | Measure behavior without changing runtime decisions |

Normalization owns identity. RAG owns evidence and source provenance. The LLM must not change an ingredient identity or manufacture evidence. If Gemini is unavailable, the system returns an explicit unavailable analysis instead of a mock success.
