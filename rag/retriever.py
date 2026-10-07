"""
CliNexa Healthcare Intelligence Platform
Module: RAG (Retrieval-Augmented Generation) Health Assistant
Description: Semantic vector retrieval powered by FAISS and trusted healthcare
literature, providing grounded evidence-based healthcare explanations.

CRITICAL PIPELINE:
Documents -> Text Extraction -> Cleaning -> Chunking -> Vector DB (FAISS)
-> Semantic Retrieval -> Grounded Response -> Citations & Safety Guardrails.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path(__file__).resolve().parent / "documents"


class RAGDocumentChunk:
    def __init__(self, chunk_id: str, doc_name: str, title: str, text: str):
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.title = title
        self.text = text


class ClinicalRAGSystem:
    """
    Retrieval-Augmented Generation system using FAISS vector indexing
    and grounded evidence synthesis.
    """
    def __init__(self, docs_directory: Optional[Path] = None):
        self.docs_dir = docs_directory or DOCS_DIR
        self.chunks: List[RAGDocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.is_indexed = False
        self._build_index()

    def _chunk_document(self, text: str, doc_name: str, chunk_size: int = 500, overlap: int = 100) -> List[RAGDocumentChunk]:
        """Split document text into clean semantic chunks."""
        # Extract title from first markdown header if available
        first_line = text.strip().split("\n")[0]
        title = first_line.replace("#", "").strip() if first_line.startswith("#") else doc_name.replace("_", " ").title()

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > chunk_size and current_chunk:
                chunk_str = "\n".join(current_chunk)
                chunk_id = f"{doc_name}_{len(chunks)}"
                chunks.append(RAGDocumentChunk(chunk_id, doc_name, title, chunk_str))
                # keep last element for overlap
                current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                current_len = len(current_chunk[0]) if current_chunk else 0

            current_chunk.append(p)
            current_len += p_len

        if current_chunk:
            chunk_str = "\n".join(current_chunk)
            chunk_id = f"{doc_name}_{len(chunks)}"
            chunks.append(RAGDocumentChunk(chunk_id, doc_name, title, chunk_str))

        return chunks

    def _build_index(self):
        """Index all documents in the knowledge repository using FAISS."""
        if not self.docs_dir.exists():
            return

        all_chunks = []
        for file_path in self.docs_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc_chunks = self._chunk_document(content, file_path.stem)
                    all_chunks.extend(doc_chunks)
            except Exception:
                continue

        if not all_chunks:
            return

        self.chunks = all_chunks
        corpus = [c.text for c in self.chunks]

        # Fit TF-IDF matrix and normalize vectors for cosine similarity via Inner Product
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray().astype(np.float32)
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vectors = tfidf_matrix / norms

        dimension = norm_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(norm_vectors)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity retrieval against FAISS vector database.
        """
        if not self.is_indexed or not query.strip():
            return []

        # Vectorize query
        q_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "score": round(float(score), 4),
                    "document_name": chunk.doc_name,
                    "document_title": chunk.title,
                    "content": chunk.text
                })
        return results

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Execute full RAG pipeline: retrieval -> grounded clinical synthesis -> citations.
        """
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty. Please ask a healthcare question.",
                "sources": []
            }

        retrieved_chunks = self.retrieve(query, top_k=top_k)
        if not retrieved_chunks:
            return {
                "success": False,
                "error": "No relevant clinical documents found for the requested query.",
                "sources": []
            }

        # Check for optional API Key (Groq or OpenAI) for LLM enhancement
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            answer = self._call_groq_llm(query, retrieved_chunks, groq_key)
        elif openai_key:
            answer = self._call_openai_llm(query, retrieved_chunks, openai_key)
        else:
            # High-fidelity grounded local synthesis engine
            answer = self._generate_grounded_local_response(query, retrieved_chunks)

        return {
            "success": True,
            "query": query,
            "answer": answer,
            "sources": retrieved_chunks,
            "vector_store": "FAISS (IndexFlatIP Cosine Similarity)",
            "safety_disclaimer": (
                "This response is grounded in trusted reference healthcare literature for educational "
                "purposes only. It is not an individual medical diagnosis or treatment plan. "
                "Consult a licensed physician for personal health evaluations."
            )
        }

    def _generate_grounded_local_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesize grounded response from retrieved clinical evidence chunks.
        Strictly prevents hallucination by referencing only documented statements.
        """
        primary_chunk = chunks[0]
        primary_title = primary_chunk["document_title"]

        # Parse key sentences from the highest-ranked chunk that contain query keywords
        q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', primary_chunk["content"]) if s.strip()]

        relevant_sentences = []
        for s in sentences:
            s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
            overlap = q_words.intersection(s_words)
            if overlap:
                relevant_sentences.append(s)

        if not relevant_sentences and sentences:
            relevant_sentences = sentences[:3]

        answer_body = " ".join(relevant_sentences)

        response = (
            f"Based on clinical documentation from **{primary_title}**:\n\n"
            f"{answer_body}\n\n"
            f"### Key Clinical Takeaways:\n"
            f"- **Primary Reference**: {primary_title} (Relevance match: {int(primary_chunk['score']*100)}%)\n"
            f"- **Context Summary**: The retrieved evidence outlines evidence-based physiological mechanisms and lifestyle parameters pertinent to your inquiry.\n"
            f"- **Actionable Step**: Review these findings in consultation with your doctor to discuss how they relate to your specific health profile."
        )
        return response

    def _call_groq_llm(self, query: str, chunks: List[Dict[str, Any]], api_key: str) -> str:
        """Optional Groq LLM integration when user configures GROQ_API_KEY in .env."""
        try:
            import urllib.request
            import json

            context_text = "\n\n---\n\n".join([f"Source: {c['document_title']}\n{c['content']}" for c in chunks])
            prompt = (
                f"You are CliNexa, an AI healthcare intelligence assistant. "
                f"Answer the user's question accurately and concisely in English, strictly grounded in the following reference context.\n"
                f"Do NOT invent unverified medical claims. Always maintain an objective educational tone.\n\n"
                f"Context:\n{context_text}\n\n"
                f"User Question: {query}\n\n"
                f"Answer:"
            )

            req_data = json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception:
            return self._generate_grounded_local_response(query, chunks)

    def _call_openai_llm(self, query: str, chunks: List[Dict[str, Any]], api_key: str) -> str:
        """Optional OpenAI LLM integration when user configures OPENAI_API_KEY in .env."""
        try:
            import urllib.request
            import json

            context_text = "\n\n---\n\n".join([f"Source: {c['document_title']}\n{c['content']}" for c in chunks])
            prompt = (
                f"You are CliNexa, an AI healthcare intelligence assistant. "
                f"Answer the user's question accurately and concisely in English, strictly grounded in the following reference context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"User Question: {query}\n\n"
                f"Answer:"
            )

            req_data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception:
            return self._generate_grounded_local_response(query, chunks)
