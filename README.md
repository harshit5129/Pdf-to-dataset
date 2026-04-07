# Pdf-to-dataset

A simple PDF-to-question dataset generator using OpenRouter AI.

## Environment variables

Create a `.env` file in the repository root or export environment variables before running.

Required:
- `OLLAMA_API_KEY` — your Ollama API key

Optional:
- `OLLAMA_BASE_URL` — Ollama host URL (default: `https://ollama.com`)
- `OLLAMA_MODEL` — model to use (default: `gpt-oss:120b`)
- `QUESTIONS_PER_CHUNK` — number of AI questions to generate per text chunk (default: `8`)

If `OPENAI_API_KEY` is already set, the script will also use that value as a fallback for `OLLAMA_API_KEY`.

Example `.env` file:

```
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=kimi-k2.5:cloud
QUESTIONS_PER_CHUNK=8
```

## Usage

```bash
python3 pdf_to_dataset.py path/to/your.pdf
```

The script will generate `dataset.json` and `dataset_pretty.json` in the repository root.
