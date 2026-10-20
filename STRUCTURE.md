                         ┌───────────────────────────┐
                         │         USER UI           │
                         │   Chatbot Web Interface   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      USER QUESTION        │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │ CONVERSATION CONTEXT PROCESSING  │
                    │                                  │
                    │ • Chat History                   │
                    │ • Follow-up Question Detection   │
                    │ • Query Contextualization        │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────┐
                    │     STANDALONE USER QUERY        │
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
             ┌─────────────────────────────────────────────┐
             │           QUERY ROUTER / RETRIEVER          │
             └───────────────────┬─────────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌────────────────────┐         ┌────────────────────┐
       │ SEMANTIC SEARCH    │         │ KEYWORD SEARCH     │
       │                    │         │                    │
       │ Hugging Face       │         │ BM25               │
       │ Embeddings         │         │ Retrieval          │
       └─────────┬──────────┘         └─────────┬──────────┘
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
                    ┌───────────────────────────┐
                    │       HYBRID SEARCH       │
                    │                           │
                    │ Merge + Rank Results      │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │        RERANKER           │
                    │ Hugging Face CrossEncoder │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ TOP RELEVANT DOCUMENTS    │
                    │ + SOURCE METADATA         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       GROQ LLM API        │
                    │                           │
                    │ Context + Question        │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     FINAL RESPONSE        │
                    │                           │
                    │ • Answer                  │
                    │ • Source Citations        │
                    │ • PDF / Web Sources       │
                    └───────────────────────────┘