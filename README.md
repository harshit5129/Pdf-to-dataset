# Pdf-to-dataset

A simple PDF-to-question dataset generator using OpenRouter AI.

## Environment variables

Create a `.env` file in the repository root or export environment variables before running.

Required:
- `OPENROUTER_API_KEY` — your OpenRouter API key

Optional:
- `OPENROUTER_MODEL` — model to use (default: `qwen/qwen3.6-plus:free`)
- `QUESTIONS_PER_CHUNK` — number of AI questions to generate per text chunk (default: `8`)

Example `.env` file:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=qwen/qwen3.6-plus:free
QUESTIONS_PER_CHUNK=8
```

## Usage

```bash
python3 pdf_to_dataset.py path/to/your.pdf
```

The script will generate `dataset.json` and `dataset_pretty.json` in the repository root.
