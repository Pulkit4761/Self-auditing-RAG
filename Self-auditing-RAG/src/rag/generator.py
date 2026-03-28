from openai import OpenAI
from . import config

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information to "
    "answer, say so. Do not use any outside knowledge."
)

class Generator:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, query: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        response = self._client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer:"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def revise(
        self, query: str, context_chunks: list[str], unsupported: list[str]
    ) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        unsupported_text = "\n".join(f"- {s}" for s in unsupported)
        response = self._client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "A previous answer contained these unsupported claims "
                        "that are NOT backed by the context:\n"
                        f"{unsupported_text}\n\n"
                        "Write a new answer using ONLY information from the "
                        "context. Do not include the unsupported claims."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
