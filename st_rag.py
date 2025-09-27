"""
st_rag.py — Lean RAG pipeline tuned for short, fact-style Q&A corpora.

Defaults:
- Chunk size: 300 (≈20% overlap by default)
- Retriever: Similarity (k=5). Set use_mmr=True to enable MMR.
- Answering: context-only, concise, with (Sec X) inline citations.

Usage:
    from st_rag import BestRAG
    rag = BestRAG.from_texts(["doc 1 text", "doc 2 text"])
    print(rag.ask("Your question"))
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------- Small, focused helpers ----------

def _extract_keywords(q: str) -> List[str]:
    """Crude content-word extraction for quick filtering/metrics."""
    return re.findall(r"[A-Za-z]{4,}", q.lower())


def _filter_support_docs(question: str, docs_list: List[Document], top_n: int = 3) -> List[Document]:
    """
    Keep docs that match any keyword from the question; fallback to original order.
    Keeps at most top_n items to reduce context noise.
    """
    keys = set(_extract_keywords(question))
    support = [d for d in docs_list if any(k in d.page_content.lower() for k in keys)]
    return (support or docs_list)[:top_n]


def _format_docs(docs_list: List[Document]) -> str:
    """
    Add lightweight citations based on `section` or `page` metadata.
    Falls back to 'Sec ?' if not present.
    """
    def sec_of(d: Document) -> str:
        sec = d.metadata.get("section")
        if sec is None:
            sec = d.metadata.get("page")
        return str(sec) if sec is not None else "?"
    return "\n\n".join([f"[Sec {sec_of(d)}] {d.page_content}" for d in docs_list])


# ---------- Core builder ----------

class BestRAG:
    """
    Build once, ask many times.

    Convenience:
        - BestRAG.from_docs(docs, ...)
        - BestRAG.from_texts(texts, metadatas=[...], ...)
        - ask(question) -> str
        - ask_return_sources(question) -> (answer, support_docs)
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self._prompt = ChatPromptTemplate.from_template(
            "Answer ONLY using the CONTEXT. Be concise. Add a citation like (Sec X) "
            "after each bullet or sentence. If the answer is not in the context, say "
            "'Not enough info'.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
        )
        self._chain = self._prompt | self.llm | StrOutputParser()

    # -------- Builders --------

    @classmethod
    def from_docs(
        cls,
        docs: List[Document],
        chunk_size: int = 300,
        chunk_overlap: Optional[int] = None,
        k: int = 5,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        use_mmr: bool = False,
        mmr_fetch_k: int = 20,
        mmr_lambda: float = 0.8,
    ) -> "BestRAG":
        """Build a FAISS index and a retriever from Documents."""
        if not docs:
            raise ValueError("from_docs: empty docs list.")

        if chunk_overlap is None:
            chunk_overlap = max(40, int(chunk_size * 0.2))  # ~20% overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("from_docs: no chunks were produced. Check your inputs.")

        emb = OpenAIEmbeddings(model=embedding_model)
        vs = FAISS.from_documents(chunks, emb)

        if use_mmr:
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": mmr_fetch_k, "lambda_mult": mmr_lambda},
            )
        else:
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

        llm = ChatOpenAI(model=chat_model, temperature=temperature)
        return cls(retriever=retriever, llm=llm)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> "BestRAG":
        """Convenience constructor when you only have raw texts."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas)]
        return cls.from_docs(docs, **kwargs)

    # -------- QA methods --------

    def ask(self, question: str) -> str:
        """Retrieve → filter to support docs → answer with citations."""
        docs_k = self.retriever.invoke(question)
        if not docs_k:
            return "Not enough info"
        support_docs = _filter_support_docs(question, docs_k, top_n=3)
        context = _format_docs(support_docs)
        return self._chain.invoke({"context": context, "question": question})

    def ask_return_sources(self, question: str) -> Tuple[str, List[Document]]:
        """Like ask(), but also return the support docs used for context."""
        docs_k = self.retriever.invoke(question)
        if not docs_k:
            return "Not enough info", []
        support_docs = _filter_support_docs(question, docs_k, top_n=3)
        context = _format_docs(support_docs)
        ans = self._chain.invoke({"context": context, "question": question})
        return ans, support_docs


if __name__ == "__main__":
    # Quick smoke test
    texts = [
        "Section 1: The Earth orbits the Sun once every 365 days.",
        "Section 2: The Moon orbits the Earth about every 27 days.",
    ]
    metas = [{"section": 1}, {"section": 2}]
    rag = BestRAG.from_texts(texts, metadatas=metas)
    print(rag.ask("How long is an Earth year?"))
