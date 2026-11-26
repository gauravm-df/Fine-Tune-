# Architecture Diagrams & System Flow
**Project:** Hybrid RAG + Fine-Tuned Foundation LLM + Exact KYC System
**Target:** Existing_LLM_Project

---

## 1. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientLayer ["1. Client & Presentation Layer"]
        User(["👤 User Query"])
        CLI["CLI Terminal (cli_chat.py)"]
        WebUI["Streamlit Interface (app_ui.py)"]
        APIClient["REST API Client / Frontend"]
    end

    subgraph OrchestrationLayer ["2. Central Orchestration & Routing Layer"]
        FastAPIServer["FastAPI Gateway (api_service.py)"]
        IntentRouter{"Intent Classifier & Query Router\n(intent_router.py)"}
        PromptManager["Route-Specific System Prompts\n(prompt_templates.py)"]
        ContextBuilder["Grounded Context Builder\n(context_builder.py)"]
    end

    subgraph DataRetrievalLayer ["3. Deterministic Knowledge & Retrieval Layer"]
        subgraph KYC_Engine ["Authoritative KYC Engine"]
            SQLGuard["SQL Sanitizer & RBAC Guard"]
            SQLiteDB[("SQLite 100K Citizen DB\n(kyc_index.sqlite)")]
            AuditLogger[("Audit Log Database")]
        end
        
        subgraph Hybrid_RAG ["Hybrid RAG Engine"]
            DocIngest["Document Chunker & Parser\n(DOCX, PDF, TXT)"]
            DenseVector[("Vector Store / Embeddings")]
            BM25Index[("BM25 Keyword Index")]
            RRF_Fusion["Reciprocal Rank Fusion\n(RRF & Reranker)"]
        end
    end

    subgraph GenerationLayer ["4. Foundation LLM Generation Layer"]
        FineTunedLLM["Qwen 2.5 (0.5B / 1.5B) + LoRA Adapter\n(Conversational Persona)"]
        OutputValidator["Anti-Hallucination & Citation Validator"]
    end

    User --> CLI & WebUI & APIClient
    CLI & WebUI & APIClient --> FastAPIServer
    FastAPIServer --> IntentRouter

    %% Routing Decisions
    IntentRouter -- "Route: EXACT_KYC" --> SQLGuard
    IntentRouter -- "Route: DOC_RAG" --> RRF_Fusion
    IntentRouter -- "Route: GENERAL_CHAT" --> ContextBuilder
    IntentRouter -- "Route: HYBRID" --> SQLGuard & RRF_Fusion

    %% Knowledge Gathering
    SQLGuard --> SQLiteDB --> ContextBuilder
    SQLGuard --> AuditLogger
    DocIngest --> DenseVector & BM25Index --> RRF_Fusion --> ContextBuilder

    %% Assembly & Inference
    PromptManager --> ContextBuilder
    ContextBuilder --> FineTunedLLM
    FineTunedLLM --> OutputValidator
    OutputValidator --> FastAPIServer
    FastAPIServer --> User
```

---

## 2. Sequence Diagram: Exact KYC vs. Document RAG vs. Conversational

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant Router as Intent Router
    participant SQL as KYC SQL Engine
    participant RAG as Hybrid RAG Engine
    participant Builder as Context Builder
    participant LLM as Fine-Tuned LLM

    User->>Router: "What is the Aadhaar for person ID P00001?"
    Router->>SQL: Execute parameterized query: SELECT * FROM citizens WHERE person_id = 'P00001'
    SQL-->>Builder: Authorized record: {name: 'Arjun Sharma', aadhaar: '997228186620', ...}
    Builder->>LLM: Strict Context: "State only the retrieved record facts."
    LLM-->>User: "The Aadhaar number for Arjun Sharma (P00001) is 997228186620."

    User->>Router: "What is the policy for data retention?"
    Router->>RAG: Hybrid Search (Vector + BM25) for 'data retention policy'
    RAG-->>Builder: Top chunks from Technical_Guide.docx with citations
    Builder->>LLM: Grounded Context with source chunks
    LLM-->>User: "According to Technical_Guide.docx [Page 3], data is retained for 7 years."

    User->>Router: "Hey! Any plans for dinner tonight?"
    Router->>Builder: Select conversational chat prompt
    Builder->>LLM: Direct prompt with conversation history
    LLM-->>User: "How about vegetable biryani? I know a great spot near the station!"
```

