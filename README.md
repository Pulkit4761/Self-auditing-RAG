# Self-auditing-RAG

README: Self-Auditing RAG System to Detect and Reduce Hallucinations

PROJECT OVERVIEW
This project implements a Self-Auditing Retrieval-Augmented Generation (RAG) system designed to detect and reduce hallucinations in Large Language Model (LLM) outputs.

Traditional RAG systems retrieve relevant documents and generate responses but do not verify whether the generated answer is actually supported by the retrieved evidence. This project introduces a faithfulness auditing layer that evaluates generated responses before returning them to the user.

PROBLEM STATEMENT
Large Language Models frequently produce hallucinated outputs, i.e., responses that appear plausible but are not grounded in factual evidence.

While RAG improves grounding by incorporating retrieved documents, it still suffers from:
- unsupported claims in generated answers
- misinterpretation of retrieved context
- mixing parametric knowledge with retrieved evidence

There is currently no built-in verification step in standard RAG pipelines.

PROPOSED SOLUTION
We introduce a Self-Auditing RAG pipeline that verifies generated answers before returning them.

Pipeline:
User Query → Retriever → Generator → Faithfulness Auditor → Decision Module → Accept / Revise / Reject

CORE COMPONENTS

1. Retriever
- Converts documents into embeddings
- Stores them in a vector database
- Retrieves top-k relevant chunks

2. Generator
- Generates candidate response using retrieved context
- Output is not trusted yet

3. Faithfulness Auditor
- Splits answer into sentences
- Compares each sentence with retrieved context
- Computes similarity score
- Identifies unsupported claims

4. Decision Module
- Accept / Revise / Reject based on faithfulness score

5. Revision Module
- Rewrites answer using only verified context
- Removes unsupported claims

KEY CONCEPT
Faithfulness = Answer supported by retrieved documents
(Not real-world truth verification)

EXAMPLE
Query: When was the first iPhone released?
Context: iPhone released in 2007
Generated: 2012 → Flagged → Revised → 2007

TECH STACK
- Embeddings: Sentence Transformers
- Vector DB: FAISS 
- LLM: OpenAI 
- Backend: Python

EVALUATION METRICS
- Faithfulness score
- Hallucination rate
- Response reliability

SCOPE
- Document QA system
- No internet fact-checking
- Works on retrieved documents only

FUTURE IMPROVEMENTS
- Explainability
- Multi-hop reasoning
- Better revision strategies

KEY INSIGHT
Retrieval alone does not guarantee correctness. Verification is required.

ONE-LINE SUMMARY
A RAG system that does not trust its own answers and verifies them before responding.

