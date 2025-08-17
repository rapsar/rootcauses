import os
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import torch

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from openai import OpenAI

from sentence_transformers import SentenceTransformer


class PhysicsRAGComparer:
    def __init__(self,
                 vocab_file: str,
                 embedding_model:   str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model:         str = "meta-llama/Llama-3.2-1B",
                 eval_model:        str = "gpt-4.1"):
        """
        Compare raw LLM vs. Retrieval-Augmented answers on physics vocab.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model, device=self.device)
        print(f"Loaded embedding model: {embedding_model}")

        # LLM
        self.llm = HuggingFaceLLM(
            model_name=llm_model,
            tokenizer_name=llm_model,
            device_map=self.device,
            generate_kwargs={"temperature": 0.7, "do_sample": True},
        )
        print(f"Loaded LLM model: {llm_model}")
        
        # Evaluation model
        self.eval_model = eval_model
        print(f"Evaluation model: {eval_model}")

        # Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=50)

        # Load vocabulary
        self.documents = self._load_vocab(vocab_file)

        # Build index
        self.index = VectorStoreIndex.from_documents(self.documents, show_progress=True)
        self.query_engine = self.index.as_query_engine(similarity_top_k=3, response_mode="compact")

    def _load_vocab(self, vocab_file: str) -> List[Document]:
        """Load physics items into LlamaIndex documents"""
        documents = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                concept = obj.get("item", "Unknown")
                description = obj.get("description", "")
                content = f"**{concept}**\n\n{description}"
                documents.append(Document(text=content, metadata={"concept": concept, "domain": "physics"}))
        print(f"Loaded {len(documents)} concepts from {vocab_file}")
        return documents

    def ask(self, question: str) -> Dict:
        """Ask with and without RAG"""
        print("\n========================================")
        print(f"QUESTION: {question}")
        print("========================================")

        # Raw LLM output (no retrieval)
        llm_prompt = f"You are a physics expert. Answer concisely:\n\nQuestion: {question}"
        llm_answer = self.llm.complete(llm_prompt).text

        # Retrieval-augmented answer
        rag_prompt = f"You are a physics expert. Use the provided physics concepts as context:\n\nQuestion: {question}"
        rag_response = self.query_engine.query(rag_prompt)

        sources = []
        if hasattr(rag_response, 'source_nodes'):
            for node in rag_response.source_nodes:
                sources.append({
                    'concept': node.metadata.get('concept', 'Unknown'),
                    'relevance': getattr(node, 'score', 0.0),
                    'preview': node.text[:150] + "..." if len(node.text) > 150 else node.text
                })

        return {
            "question": question,
            "llm_answer": llm_answer.strip(),
            "rag_answer": str(rag_response),
            "rag_sources": sources
        }

    def evaluate_answers(self, question: str, llm_answer: str, rag_answer: str, api_key=None) -> str:
        if self.eval_model is None:
            return "skipped"
        
        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are an evaluator model. Given the question and two answers, "
            f"decide which answer is better.\n\n"
            f"Question: {question}\n\n"
            f"Answer 1 (LLM): {llm_answer}\n\n"
            f"Answer 2 (RAG): {rag_answer}\n\n"
            f"Respond strictly with JSON in the format {{\"best\": \"llm\"}} or {{\"best\": \"rag\"}}."
        )
        response = client.chat.completions.create(
            model=self.eval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        try:
            result = json.loads(content)
            best = result.get("best", "").lower()
            if best in ["llm", "rag"]:
                return best
        except json.JSONDecodeError:
            pass
        return "unknown"


def main():
    vocab_file = "data/physics_vocabulary_with_attributes_and_description.jsonl"
    # embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = "../rootc/physics-embedding-model-20250816_192016"
    llm_model = "HuggingFaceTB/SmolLM3-3B" #"HuggingFaceTB/SmolLM2-1.7B-Instruct" #"HuggingFaceTB/SmolLM3-3B"
    eval_model = "gpt-4.1"

    comparer = PhysicsRAGComparer(
        vocab_file=vocab_file, 
        embedding_model = embedding_model,
        llm_model=llm_model, 
        eval_model=eval_model)

    questions = [
        "Why does a ball bounce lower each time it hits the ground? Find all related physics concepts and variables.",
        "How do rockets work in space where there's no air to push against? Find all related physics concepts and variables.",
        "What makes quantum computers potentially more powerful than classical computers? Find all related physics concepts and variables.",
        "Why can't we build a machine that runs forever without energy input? Find all related physics concepts and variables.",
        "How does GPS account for time running differently in space? Find all related physics concepts and variables."
    ]

    results = []
    for q in questions:
        res = comparer.ask(q)
        print("\n--- LLM ANSWER ---")
        print(res["llm_answer"])
        print("\n--- RAG ANSWER ---")
        print(res["rag_answer"])
        print(f"\n--- RAG SOURCES ({len(res['rag_sources'])}) ---")
        for src in res["rag_sources"]:
            print(f"- {src['concept']} (relevance: {src['relevance']:.3f})")

        if comparer.eval_model is not None:
            decision = comparer.evaluate_answers(q, res["llm_answer"], res["rag_answer"])
            print(f"\nEVALUATOR DECISION: {decision.upper()}")
        else:
            print("\nEVALUATOR DECISION: SKIPPED")
        print("========================================")

        results.append(res)

    return comparer, results


if __name__ == "__main__":
    comparer, results = main()