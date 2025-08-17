from openai import OpenAI
import json
from typing import List
import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

# --- Vocab Loader ---
def load_vocab(vocab_file: str) -> List[Document]:
    """Load physics items into LlamaIndex documents from a JSONL file"""
    documents = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            concept = obj.get("item", "Unknown")
            description = obj.get("description", "")
            content = f"**{concept}**\n\n{description}"
            documents.append(Document(text=content, metadata={"concept": concept, "domain": "physics"}))
    return documents

# --- Helper: Evaluate answers using OpenAI GPT-4.1 ---
def evaluate_answers(question: str, llm_answer: str, rag_answer: str, api_key=None) -> str:
    """
    Use OpenAI GPT-4.1 to compare LLM and RAG answers, return 'llm', 'rag', or 'unknown'.
    """
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
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        best = result.get("best", "").lower()
        if best in ["llm", "rag"]:
            return best
    except Exception:
        pass
    return "unknown"

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Settings")

llm_choice = st.sidebar.selectbox(
    "Choose local LLM model:",
    [
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "HuggingFaceTB/SmolLM3-3B"
    ]
)

embed_choice = st.sidebar.selectbox(
    "Choose embedding model:",
    [
        "all-MiniLM-L6-v2",
        # "../rootc/physics-embedding-model-20250816_192016",
        "physics-embedding-model-large-20250817_195614",
        "../rootc/another-checkpoint"
    ]
)

# Clear session state if model selections changed
if 'prev_llm_choice' not in st.session_state:
    st.session_state.prev_llm_choice = llm_choice
if 'prev_embed_choice' not in st.session_state:
    st.session_state.prev_embed_choice = embed_choice

if (st.session_state.prev_llm_choice != llm_choice or 
    st.session_state.prev_embed_choice != embed_choice):
    # Clear cached answers when models change
    st.session_state.llm_answer = None
    st.session_state.rag_answer = None
    if hasattr(st.session_state, 'source_nodes'):
        delattr(st.session_state, 'source_nodes')
    st.session_state.prev_llm_choice = llm_choice
    st.session_state.prev_embed_choice = embed_choice

# --- Question Input ---
st.title("🔎 RAG Comparison Playground")
question = st.text_input("Ask a question:")

# Initialize session state
if 'llm_answer' not in st.session_state:
    st.session_state.llm_answer = None
if 'rag_answer' not in st.session_state:
    st.session_state.rag_answer = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None

# --- Load embedding model ---
embed_model = HuggingFaceEmbedding(model_name=embed_choice)

# --- Load LLM locally ---
device = "cuda" if torch.cuda.is_available() else "cpu"
llm = HuggingFaceLLM(
    model_name=llm_choice,
    device_map=device,   # will use GPU if available
    tokenizer_name=llm_choice,
    max_new_tokens=128,
    context_window=2048,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    model_kwargs={
        "torch_dtype": torch.float16,  
    },
)

# --- Dummy docs (replace with your own corpus or loaders) ---
vocab_file = "./data/physics_vocabulary_with_attributes_and_description.jsonl"
documents = load_vocab(vocab_file)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

# --- Run comparison ---
if question:
    # Check if question changed, if so, clear previous answers
    if st.session_state.current_question != question:
        st.session_state.llm_answer = None
        st.session_state.rag_answer = None
        st.session_state.current_question = question
    
    # Generate responses only if not already generated
    if st.session_state.llm_answer is None:
        st.subheader("LLM Response (No RAG)")
        with st.spinner("Generating LLM response..."):
            llm_prompt = f"You are a physics expert. Answer concisely:\n\nQuestion: {question}"
            no_rag_response = llm.complete(llm_prompt)
            st.session_state.llm_answer = no_rag_response.text.strip()
    else:
        st.subheader("LLM Response (No RAG)")
    
    st.write(st.session_state.llm_answer)

    if st.session_state.rag_answer is None:
        st.subheader("LLM Response (With RAG)")
        with st.spinner("Generating RAG response..."):
            rag_response = query_engine.query(question)
            st.session_state.rag_answer = rag_response.response
            # Store source nodes if needed
            if hasattr(rag_response, "source_nodes"):
                st.session_state.source_nodes = rag_response.source_nodes
    else:
        st.subheader("LLM Response (With RAG)")
    
    st.write(st.session_state.rag_answer)
    
    if hasattr(st.session_state, 'source_nodes'):
        st.subheader("🔎 RAG Sources")
        for node in st.session_state.source_nodes:
            st.markdown(f"- **{node.metadata.get('concept', 'Unknown')}** "
                        f"(relevance: {getattr(node, 'score', 0.0):.3f})")

    if st.session_state.llm_answer and st.session_state.rag_answer:
        st.subheader("⚖️ Evaluation")
        if st.button("Evaluate with GPT-4.1"):
            with st.spinner("Evaluating answers with GPT-4.1..."):
                api_key = None
                decision = evaluate_answers(question, st.session_state.llm_answer, st.session_state.rag_answer, api_key=api_key)
            st.write(f"**Best answer:** {decision.upper()}")