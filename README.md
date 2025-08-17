# `rootcauses`: physics-informed concept embeddings for specialized LLM physicist

Solving advanced problems in physics require to consider all concepts, variables, and laws related to the problem.

The goal of `rootcauses` is to provide embeddings of physics concepts for retrieval (eg, RAG) by an LLM to answer physics related questions.

## workflow
- generate vocabulary (constants/variables/concepts/principles) using GPT-4.1
- assemble pairs of related concepts from vocabulary (eg, related_to, more_fundamental_than)
- use vocabulary and pairs to finetune an embedding model (eg, from sentence_transformer)
- establish RAG pipeline by connecting embedding model with LLM
- evaluate LLM with and without RAG using evaluator model (eg, GPT-4.1)
