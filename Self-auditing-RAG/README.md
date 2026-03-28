# Self-Auditing RAG System

A Retrieval-Augmented Generation system with a built-in faithfulness auditing layer that detects and reduces hallucinations in LLM outputs.

## Problem Statement

Large Language Models frequently produce **hallucinated outputs** — responses that sound plausible but are not grounded in factual evidence. Standard RAG pipelines improve grounding by incorporating retrieved documents, but they still suffer from:

- Unsupported claims sneaking into generated answers
- Misinterpretation of retrieved context
- The LLM mixing its own parametric knowledge with retrieved evidence

There is no built-in verification step in a standard RAG pipeline. This project solves that.

## How It Works

The system adds a **faithfulness auditing layer** between generation and output. Instead of blindly trusting the LLM's response, every answer is verified against the retrieved context before being shown to the user.

### Pipeline

```
User Query
    │
    ▼
┌──────────┐
│ Retriever │  ── Embeds query, searches FAISS index, returns top-k chunks
└────┬─────┘
     │
     ▼
┌───────────┐
│ Generator │  ── Sends query + context to gpt-4o-mini, gets candidate answer
└────┬──────┘
     │
     ▼
┌───────────────────┐
│ Faithfulness      │  ── Splits answer into sentences, scores each against
│ Auditor           │     retrieved context using cosine similarity
└────┬──────────────┘
     │
     ▼
┌─────────────────┐
│ Decision Module │  ── Accept / Revise / Reject based on faithfulness score
└────┬────────────┘
     │
     ├── ACCEPT  (score ≥ 0.6)  →  Return answer as-is
     ├── REVISE  (0.4 ≤ score < 0.6)  →  Re-generate without unsupported claims
     └── REJECT  (score < 0.4)  →  Tell user the answer can't be verified
```

### Step-by-Step Breakdown

#### 1. Document Ingestion (`src/rag/retriever.py`)
- Reads all `.txt` files from the `documents/` folder
- Splits each document into overlapping chunks (500 characters, 50-character overlap)
- Converts chunks into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
- Normalizes embeddings and stores them in a FAISS inner-product index
- Saves the index and chunk metadata to disk (`index/` folder) for reuse
`
#### 2. Retrieval (`src/rag/retriever.py`)
- Embeds the user's query using the same Sentence Transformer model
- Searches the FAISS index for the top-5 most similar chunks (configurable)
- Returns these chunks as context for the generator

#### 3. Generation (`src/rag/generator.py`)
- Sends the query along with retrieved context chunks to OpenAI's `gpt-4o-mini`
- The system prompt explicitly instructs the model to answer **only** from the provided context
- Uses low temperature (0.3) to reduce creative/hallucinated outputs
- The generated answer is treated as a **candidate** — not yet trusted

#### 4. Faithfulness Auditing (`src/rag/auditor.py`)
This is the core innovation of the project:

- **Sentence splitting**: The candidate answer is split into individual sentences
- **Embedding**: Each sentence is embedded using the same Sentence Transformer model
- **Cosine similarity**: Each sentence embedding is compared against all retrieved chunk embeddings
- **Per-sentence score**: The maximum similarity between a sentence and any context chunk becomes that sentence's faithfulness score
- **Verdict**: Sentences scoring below 0.5 are flagged as **unsupported**
- **Overall score**: The mean of all per-sentence scores gives the overall faithfulness score

#### 5. Decision (`src/rag/decision.py`)
Based on the overall faithfulness score:

| Score Range | Decision | Action |
|---|---|---|
| ≥ 0.6 | **ACCEPT** | Return the answer as-is |
| 0.4 – 0.6 | **REVISE** | Re-generate, explicitly excluding unsupported claims |
| < 0.4 | **REJECT** | Inform user the answer couldn't be verified |

#### 6. Revision (`src/rag/generator.py`)
When the decision is REVISE:
- The unsupported claims are sent back to the LLM
- The LLM is instructed to write a new answer using **only** the verified context
- The revision uses an even lower temperature (0.2) for tighter grounding
- Only one revision attempt is made to conserve API budget

## Key Concept

**Faithfulness ≠ Factual correctness**

This system checks whether the answer is **supported by the retrieved documents**, not whether it's true in the real world. If the documents contain incorrect information, the system will still accept answers grounded in those documents. The goal is to eliminate claims the LLM invented beyond what the context provides.

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Converts text to 384-dimensional vectors. Runs locally, no API cost |
| Vector Database | `faiss-cpu` | Fast similarity search over document embeddings |
| LLM | OpenAI `gpt-4o-mini` | Generates answers and revisions. Chosen for low cost (~$0.15/1M input tokens) |
| API Client | `openai` | Communicates with OpenAI's API |
| Environment | `python-dotenv` | Loads API key from `.env` file |
| CLI Output | `rich` | Colored terminal output with tables and panels |
| Math | `numpy` | Array operations for embedding similarity computation |
| Package Manager | `uv` | Fast Python package and project management |

## Project Structure

```
self-auditing-rag/
├── main.py                 # CLI entry point (ingest / query commands)
├── pyproject.toml           # Project config and dependencies
├── .env                     # OpenAI API key (not committed to git)
├── documents/               # Place your .txt files here
├── index/                   # Auto-generated FAISS index and chunk metadata
└── src/rag/
    ├── config.py            # All settings and thresholds
    ├── embedder.py          # Sentence Transformer wrapper
    ├── retriever.py         # Document chunking, FAISS indexing, retrieval
    ├── generator.py         # OpenAI API calls for generation and revision
    ├── auditor.py           # Faithfulness scoring engine
    ├── decision.py          # Accept / Revise / Reject logic
    └── pipeline.py          # Orchestrates the full pipeline
```

## Setup and Usage

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key

### Installation

```bash
uv sync
```

### Configuration

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

### Running

**1. Add documents** — place `.txt` files in the `documents/` folder.

**2. Ingest** — build the vector index:

```bash
uv run python main.py ingest
```

**3. Query** — ask questions interactively:

```bash
uv run python main.py query
```

The CLI displays:
- The final answer in a colored panel (green = accepted, yellow = revised, red = rejected)
- A faithfulness audit table showing each sentence's similarity score and verdict
- The overall faithfulness score
- Whether the answer was revised

## Example

```
Query> When was the first iPhone released?

╭─ Answer  ACCEPT ─────────────────────────────────╮
│ The first iPhone was released in 2007.            │
╰───────────────────────────────────────────────────╯

           Faithfulness Audit
┌──────────────────────────────┬────────┬───────────┐
│ Sentence                     │ Score  │ Verdict   │
├──────────────────────────────┼────────┼───────────┤
│ The first iPhone was         │ 0.812  │ Supported │
│ released in 2007.            │        │           │
└──────────────────────────────┴────────┴───────────┘
  Overall faithfulness: 0.812
```

## Configurable Thresholds

All thresholds can be adjusted in `src/rag/config.py`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per document chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| `TOP_K` | 5 | Number of chunks retrieved per query |
| `ACCEPT_THRESHOLD` | 0.6 | Minimum score to accept an answer |
| `REJECT_THRESHOLD` | 0.4 | Score below which answer is rejected |
| `SENTENCE_SUPPORT_THRESHOLD` | 0.5 | Minimum score for a sentence to be considered supported |

## Evaluation Metrics

- **Faithfulness Score**: Mean cosine similarity between answer sentences and retrieved context (0.0 – 1.0)
- **Hallucination Rate**: Proportion of sentences flagged as unsupported
- **Response Reliability**: Proportion of queries that result in ACCEPT decisions

## Limitations

- Only works with `.txt` documents (no PDF/HTML parsing)
- Faithfulness is measured via embedding similarity, not logical entailment
- No internet fact-checking — answers are verified against retrieved documents only
- Single revision attempt to conserve API budget
