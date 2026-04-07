import ast
import sys
import os
import re
import json
import fitz
from collections import Counter
import time
import random
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

random.seed(42)

# Thread-safe lock for progress tracking
progress_lock = Lock()


def load_env_file(env_path=None):
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

VALID_QUESTION_TYPES = {"comprehension", "character", "plot", "vocabulary", "true_false", "fill_blank", "open_ended", "theme"}
MIN_QUESTION_LENGTH = 10
MIN_ANSWER_LENGTH = 5


def validate_question(q):
    if not isinstance(q, dict):
        return False, "Not a dictionary"
    if "question" not in q or "answer" not in q or "type" not in q:
        return False, "Missing required fields"
    if q["type"] not in VALID_QUESTION_TYPES:
        return False, f"Invalid type: {q['type']}"
    if not isinstance(q["question"], str) or len(q["question"].strip()) < MIN_QUESTION_LENGTH:
        return False, "Question too short"
    if not isinstance(q["answer"], str) or len(q["answer"].strip()) < MIN_ANSWER_LENGTH:
        return False, "Answer too short"
    if q["question"].strip().lower() == q["answer"].strip().lower():
        return False, "Question and answer identical"
    return True, "Valid"


def validate_ai_questions(questions, chunk_idx):
    valid_questions = []
    invalid_count = 0
    for i, q in enumerate(questions):
        is_valid, reason = validate_question(q)
        if is_valid:
            valid_questions.append(q)
        else:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"    [WARN] Chunk {chunk_idx+1}, Q{i+1}: {reason}")
    if invalid_count > 3:
        print(f"    [WARN] ... and {invalid_count - 3} more invalid questions skipped")
    return valid_questions, invalid_count


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
QUESTIONS_PER_CHUNK = int(os.environ.get("QUESTIONS_PER_CHUNK", "8"))
USE_AI_GENERATION = True


def extract_text_from_pdf(pdf_path):
    print(f"\n{'='*60}")
    print(f"  PDF QUESTION GENERATOR")
    print(f"{'='*60}")
    print(f"\nOpening PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    print(f"Total pages: {num_pages}")

    page_texts: list[str] = []
    full_text: str = ""

    for i in range(num_pages):
        page = doc[i]
        text = str(page.get_text())
        page_texts.append(text)
        full_text = full_text + text + "\n"
        progress = (i + 1) / num_pages * 100
        print(f"  Extracting page {i+1}/{num_pages} [{progress:.0f}%]", end='\r')

    print(f"\n  Extraction complete! Total characters: {len(full_text)}")
    doc.close()
    return full_text, page_texts, num_pages


def extract_novel_title(pdf_path):
    basename = os.path.basename(pdf_path)
    title = os.path.splitext(basename)[0]
    title = re.sub(r'[_-]+', ' ', title)
    return title.strip()


def split_into_chunks(text, min_chunk_size=500, max_chunk_size=1500):
    chapter_pattern = r'(?i)(?:chapter|part|prologue|epilogue|volume)\s+\d+[a-z]*\s*:?\s*'

    chapters = re.split(chapter_pattern, text)
    chapters = [c.strip() for c in chapters if len(c.strip()) > 100]

    if len(chapters) > 1:
        print(f"  Detected {len(chapters)} chapters/sections")
        return chapters

    print("  No chapter markers found, splitting by paragraphs...")
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    print(f"  Created {len(chunks)} chunks")
    return chunks


def extract_names(text):
    sentences = re.split(r'[.!?]+', text)
    names = set()

    for sentence in sentences[:50]:
        words = sentence.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2 and word.isalpha():
                if i + 1 < len(words) and words[i+1].istitle() and words[i+1].isalpha():
                    names.add(f"{word} {words[i+1]}")
                elif word not in {'The', 'This', 'That', 'There', 'Their', 'These', 'Those', 'What', 'When', 'Where', 'Which', 'While', 'With', 'Without', 'Within', 'About', 'Above', 'After', 'Before', 'Between', 'Through', 'During', 'Under', 'Again', 'Further', 'Then', 'Once', 'Here', 'All', 'Each', 'Every', 'Both', 'Few', 'More', 'Most', 'Other', 'Some', 'Such', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very', 'Can', 'Will', 'Just', 'Don', 'Now', 'And', 'But', 'For', 'Nor', 'Not', 'Or', 'Yet', 'From', 'Into', 'Onto', 'Upon', 'Toward', 'Until', 'Against', 'Among', 'Beside', 'Beyond', 'Except', 'Inside', 'Outside', 'Across', 'Around', 'Behind', 'Below', 'Beneath', 'Beside', 'Near', 'Since', 'Throughout'}:
                    names.add(word)

    return list(names)[:30]


def extract_key_sentences(text, max_sentences=20):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if 30 < len(s) < 300]

    key_sentences = []
    for s in sentences:
        if any(c.isupper() for c in s[:20]) and len(s.split()) > 8:
            key_sentences.append(s)
        if len(key_sentences) >= max_sentences:
            break

    return key_sentences


def generate_comprehension_questions(chunk, novel_title):
    questions = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    sentences = [s.strip() for s in sentences if 40 < len(s) < 250]

    templates = [
        ("What happens in the story when {context}?", "{answer}"),
        ("Describe the events where {context}", "{answer}"),
        ("According to the text, what occurs when {context}?", "{answer}"),
    ]

    for sentence in sentences[:10]:
        template = random.choice(templates)
        answer = sentence

        if len(sentence) > 80:
            question_text = sentence[:sentence.find(',', 40) if ',' in sentence[40:] else 60]
        else:
            question_text = sentence[:50]

        question = f"Based on the novel '{novel_title}', what happens in the passage that mentions: \"{question_text}...\"?"
        questions.append({
            "type": "comprehension",
            "question": question,
            "answer": answer
        })

    return questions


def generate_character_questions(chunk, novel_title, names):

    questions = []

    if not names:
        return questions

    for name in names[:8]:
        matching_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if name in s and len(s) > 30]

        if matching_sentences:
            answer = matching_sentences[0]
            templates = [
                f"Who is {name} in the novel '{novel_title}'?",
                f"What role does {name} play in the story?",
                f"Describe the character of {name} based on the text.",
                f"What does {name} do in this passage?",
            ]
            question = random.choice(templates)
            questions.append({
                "type": "character",
                "question": question,
                "answer": answer
            })

    return questions


def generate_plot_questions(chunk, novel_title):

    questions = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    action_sentences = [s.strip() for s in sentences if any(w in s.lower() for w in ['then', 'after', 'before', 'when', 'suddenly', 'finally', 'eventually', 'however', 'therefore', 'consequently']) and 40 < len(s) < 250]

    for sentence in action_sentences[:5]:
        question = f"What leads to the events described in: \"{sentence[:60]}...\"?"
        questions.append({
            "type": "plot",
            "question": question,
            "answer": sentence
        })

    return questions


def generate_vocabulary_questions(chunk, novel_title):
    """Generate vocabulary-in-context questions."""
    questions = []
    words = re.findall(r'\b[A-Za-z]{6,}\b', chunk)
    unique_words = list(set(w for w in words if w[0].islower()))[:10]

    for word in unique_words[:5]:
        matching_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if word.lower() in s.lower() and len(s) > 30]
        if matching_sentences:
            context = matching_sentences[0]
            question = f"In the context of '{novel_title}', what does '{word}' mean in this passage: \"{context[:80]}...\"?"
            questions.append({
                "type": "vocabulary",
                "question": question,
                "answer": f"In this context, '{word}' is used in the sentence: \"{context}\" The meaning relates to how it's used in this passage."
            })

    return questions


def generate_truefalse_questions(chunk, novel_title):
    
    questions = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    declarative = [s.strip() for s in sentences if 30 < len(s) < 200 and not s.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who', 'Which', 'Is ', 'Are ', 'Do ', 'Does '))]

    for sentence in declarative[:5]:
        question = f"True or False: In '{novel_title}', the following statement is accurate: \"{sentence}\""
        questions.append({
            "type": "true_false",
            "question": question,
            "answer": f"True. The text states: \"{sentence}\""
        })

    return questions


def generate_fillblank_questions(chunk, novel_title):

    questions = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    good_sentences = [s.strip() for s in sentences if 40 < len(s) < 200]

    for sentence in good_sentences[:5]:
        words = sentence.split()
        if len(words) < 6:
            continue

        blank_idx = random.randint(2, min(len(words) - 2, 6))
        blank_word = words[blank_idx]

        if len(blank_word) < 3:
            continue

        words[blank_idx] = "_____"
        blanked = ' '.join(words)

        question = f"Fill in the blank from '{novel_title}': {blanked}"
        questions.append({
            "type": "fill_blank",
            "question": question,
            "answer": f"The missing word is '{blank_word}'. Complete sentence: \"{sentence}\""
        })

    return questions


def generate_openended_questions(chunk, novel_title):
    """Generate open-ended analytical questions."""
    questions = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    meaningful = [s.strip() for s in sentences if 50 < len(s) < 250]

    if meaningful:
        sample = meaningful[0]
        templates = [
            f"Why do you think the events in '{novel_title}' unfold as described: \"{sample[:80]}...\"?",
            f"How does the passage \"{sample[:60]}...\" contribute to the overall narrative of '{novel_title}'?",
            f"What is the significance of the scene where the text describes: \"{sample[:70]}...\"?",
        ]
        question = random.choice(templates)
        questions.append({
            "type": "open_ended",
            "question": question,
            "answer": f"Based on the text: \"{sample}\" This passage contributes to the narrative by developing the story and characters within the context of {novel_title}."
        })

    return questions


def generate_ai_questions(chunk, novel_title, chunk_idx, num_questions=8):
    """Generate high-quality questions using Ollama."""
    if not OLLAMA_API_KEY:
        print(f"\n  Warning: OLLAMA_API_KEY not set. Cannot generate questions.")
        return []

    system_prompt = f"""You are an expert literary analyst and educator specializing in creating high-quality comprehension questions for novels and literature. 

Your task is to generate thoughtful, diverse questions based on excerpts from the novel '{novel_title}'. 

Create questions across these categories:
1. comprehension - Understanding plot and events
2. character - Character analysis and motivations
3. plot - Cause and effect in the narrative
4. vocabulary - Words in context
5. true_false - Factual statements about the text
6. fill_blank - Missing word from quoted text
7. open_ended - Analytical and interpretive questions
8. theme - Thematic analysis and deeper meaning

Requirements:
- Questions should be specific to the text provided
- Answers should be detailed and reference the actual text
- Vary question difficulty (basic comprehension to deep analysis)
- Make questions natural and engaging, not mechanical
- Answers should be 2-4 sentences with textual evidence

Respond ONLY with valid JSON in this exact format:
[
  {{
    "type": "comprehension",
    "question": "Your question here?",
    "answer": "Your detailed answer here."
  }}
]

Generate exactly {num_questions} questions total, with variety in types."""

    user_prompt = f"""Analyze this excerpt from the novel '{novel_title}' (Chunk {chunk_idx + 1}) and generate {num_questions} diverse comprehension questions:

---
{chunk[:3000]}
---

Generate questions that test understanding of plot, characters, themes, vocabulary, and deeper meaning. Make questions specific to this exact text."""

    try:
        client = Client(
            host=OLLAMA_BASE_URL,
            headers={"Authorization": "Bearer " + OLLAMA_API_KEY}
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = ""
        for part in client.chat(OLLAMA_MODEL, messages=messages, stream=True):
            message = part.get("message", {})
            content_part = message.get("content", "")
            if isinstance(content_part, list):
                for item in content_part:
                    if isinstance(item, dict) and item.get("type") == "text":
                        response_text += item.get("text", "")
                    elif isinstance(item, str):
                        response_text += item
            elif isinstance(content_part, dict):
                if content_part.get("type") == "text":
                    response_text += content_part.get("text", "")
                elif isinstance(content_part, str):
                    response_text += content_part
            elif isinstance(content_part, str):
                response_text += content_part

        content = response_text

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "".join(text_parts)
        elif isinstance(content, dict) and content.get("type") == "text":
            content = content.get("text", "")

        questions = None
        if isinstance(content, str):
            json_match = re.search(r'(\[[\s\S]*\])', content)
            if json_match:
                candidate = json_match.group(1)
                try:
                    questions = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        questions = ast.literal_eval(candidate)
                    except (ValueError, SyntaxError):
                        questions = None
        elif isinstance(content, list):
            questions = content

        if isinstance(questions, list):
            valid_questions, invalid_count = validate_ai_questions(questions, chunk_idx)
            print(f"\n  [AI] Chunk {chunk_idx + 1}: {len(valid_questions)} valid, {invalid_count} invalid (from {len(questions)} generated)")
            return valid_questions

        print(f"\n  [WARN] Could not parse valid AI response for chunk {chunk_idx + 1}")
        print(f"  Response preview: {content[:400]}")
        return []

    except Exception as e:
        print(f"\n  Error calling Ollama API: {e}")
        return []


def generate_questions_from_chunk(chunk, novel_title, chunk_idx):
    return generate_ai_questions(chunk, novel_title, chunk_idx, num_questions=QUESTIONS_PER_CHUNK)


def process_chunk_worker(chunk_idx, chunk, novel_title):
    """Worker function for multithreading - processes a single chunk."""
    qa = generate_questions_from_chunk(chunk, novel_title, chunk_idx)
    return chunk_idx, qa


def convert_to_gpt_format(qa_pairs, novel_title):
    dataset = []
    system_prompt = f"You are a helpful assistant analyzing the web novel '{novel_title}'. Answer questions based on the text content provided."

    for qa in qa_pairs:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]}
            ]
        }
        dataset.append(entry)

    return dataset


def display_summary(num_pages, num_chunks, qa_pairs, novel_title, elapsed_time=0.0):
    type_counts = Counter(qa["type"] for qa in qa_pairs)

    print(f"\n{'='*60}")
    print(f"  CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Novel:              {novel_title}")
    print(f"  Pages processed:    {num_pages}")
    print(f"  Text chunks:        {num_chunks}")
    print(f"  Total Q&A pairs:    {len(qa_pairs)}")
    print(f"  Generation time:    {elapsed_time:.1f}s")
    if elapsed_time > 0 and len(qa_pairs) > 0:
        print(f"  Avg per question:   {elapsed_time/len(qa_pairs):.2f}s")
    print(f"\n  Question Type Breakdown:")

    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = '#' * (count // 2)
        print(f"    {qtype:<15} {count:>3} {bar}")

    print(f"\n{'='*60}")


def display_preview(dataset, count=3):
    print(f"\n  PREVIEW (first {min(count, len(dataset))} entries):")
    print(f"  {'-'*56}")

    for i, entry in enumerate(dataset[:count]):
        print(f"\n  Entry {i+1}:")
        for msg in entry["messages"]:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 120:
                content = content[:120] + "..."
            print(f"    [{role}]: {content}")
        print(f"  {'-'*56}")


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_files[0])
        else:
            print("Error: No PDF file found. Usage: python pdf_to_dataset.py <path_to_pdf>")
            sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    novel_title = extract_novel_title(pdf_path)

    if not OLLAMA_API_KEY:
        print("Error: OLLAMA_API_KEY is required for direct AI question generation.")
        print("Set the environment variable OLLAMA_API_KEY or create a .env file with OLLAMA_API_KEY=your_key")
        sys.exit(1)

    full_text, page_texts, num_pages = extract_text_from_pdf(pdf_path)

    print(f"\nProcessing novel: {novel_title}")
    chunks = split_into_chunks(full_text)

    print(f"  AI-powered generation enabled (Model: {OLLAMA_MODEL})")
    print(f"  Questions per chunk: {QUESTIONS_PER_CHUNK}")
    print(f"  Using multithreading for parallel processing")
    print(f"\nGenerating questions from {len(chunks)} chunks...")
    all_qa = []
    start_time = time.time()
    
    max_workers = min(4, len(chunks))  # Use up to 4 threads
    chunk_results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk_worker, i, chunk, novel_title): i for i, chunk in enumerate(chunks)}
        completed_chunks = 0
        
        for future in as_completed(futures):
            chunk_idx, qa = future.result()
            chunk_results[chunk_idx] = qa
            completed_chunks += 1
            
            progress = (completed_chunks / len(chunks)) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / completed_chunks
            eta = avg_time * (len(chunks) - completed_chunks)
            print(f"  [{completed_chunks}/{len(chunks)}] {progress:.0f}% | {len(qa)} Qs | ETA: {eta:.0f}s", end='\r', flush=True)
    
    # Reconstruct results in original order
    for i in range(len(chunks)):
        if i in chunk_results:
            all_qa.extend(chunk_results[i])

    elapsed_total = time.time() - start_time
    print(f"\n  Generation complete in {elapsed_total:.1f}s")

    print(f"\n  Total questions generated: {len(all_qa)}")

    dataset = convert_to_gpt_format(all_qa, novel_title)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(script_dir, "dataset.json")
    pretty_path = os.path.join(script_dir, "dataset_pretty.json")

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(pretty_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    display_summary(num_pages, len(chunks), all_qa, novel_title, elapsed_total)
    display_preview(dataset)

    print(f"\n  Output files:")
    print(f"    JSONL: {jsonl_path}")
    print(f"    Pretty-printed JSON:   {pretty_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
