# `rootcauses`: physics-informed concept embeddings for specialized LLM physicist

Solving advanced problems in physics require to consider all concepts, variables, and laws related to the problem.

The goal of `rootcauses` is to provide embeddings of physics concepts for retrieval (eg, RAG) by an LLM to answer physics related questions.

## workflow
- generate vocabulary (constants/variables/concepts/principles) using GPT-4.1
- assemble pairs of related concepts from vocabulary (eg, related_to, more_fundamental_than)
- use vocabulary and pairs to finetune an embedding model (eg, from sentence_transformer)
- establish RAG pipeline by connecting embedding model with LLM
- evaluate LLM with and without RAG using evaluator model (eg, GPT-4.1)

### context
Consider the following physics problem:
a snapping shrimp can create cavitation bubbles by snapping its claw extremely fast.
What is the timescale of bubble collapse?

To solve this problem (which I had to do many years ago as prep for the physics olympiads), we use a simple yet insightful approach: 
identify all relevant physical quantities, then apply dimensional analysis and basic physics reasoning to construct the characteristic timescale.

In this case, the relevant variables appear to be: 
- the size of the bubble, radius $R$
- the pressure of the water outside, $P$
- the vapor pressure inside the bubble, negligible
- and the density of the surrounding water, $\rho$

Considering these parameters, its easy to see the only possible timescale is:

$$\tau \sim R \sqrt{\frac{\rho}{P}}$$

(Rayleigh collapse time, btw). Voila!

So, the goal of the physics-aware embedding model is to problem a reasoning LLM with the ability to retrieve the principles, constants and variables that are necessary to solve a given problem.


