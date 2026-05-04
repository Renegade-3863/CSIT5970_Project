# CSIT5970 Project — 智能体部署及优化

An end-to-end intelligent agent system combining **cloud-native deployment**, **agent engineering**, and **comprehensive benchmarking**.

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     Main Entry Point                      │
│                       main.py                             │
└──────────────┬───────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │   Agent Core    │  ← ReAct loop (Thought → Action → Observation)
       │  agent/core.py  │
       └──┬──────┬───────┘
          │      │
    ┌─────▼──┐  ┌▼──────────────┐
    │Persona │  │ Memory Manager │
    │Manager │  │  (RAG + STM)   │
    └────────┘  └───────┬────────┘
                        │
              ┌─────────▼──────────┐
              │    RAG System       │
              │  ┌──────────────┐  │
              │  │ VectorStore  │  │ ← ChromaDB + SentenceTransformers
              │  │  (ChromaDB)  │  │
              │  └──────────────┘  │
              │  ┌──────────────┐  │
              │  │  Indexer     │  │ ← txt / pdf / docx / md
              │  └──────────────┘  │
              └────────────────────┘
                        │
              ┌─────────▼──────────┐
              │    Tool Registry    │
              │  ┌──────────────┐  │
              │  │KnowledgeBase │  │ ← RAG-backed retrieval
              │  ├──────────────┤  │
              │  │CodeExecutor  │  │ ← Python / Bash sandbox
              │  ├──────────────┤  │
              │  │  WebSearch   │  │ ← DuckDuckGo (no API key)
              │  └──────────────┘  │
              └────────────────────┘
```

---

## 💡 Key Features

| Feature | Description |
|---|---|
| **Expert Personas** | Inject specialized system prompts (Linux expert, Security expert) |
| **ReAct Loop** | Reasoning + Acting agent loop with up to N iterations |
| **RAG Memory** | ChromaDB vector store with sentence-transformer embeddings |
| **Custom Tools** | Knowledge base retrieval, code execution, web search |
| **Benchmark Framework** | 3-mode evaluation with LLM-as-a-Judge scoring |
| **Docker Deployment** | One-command cloud deployment via Docker Compose |

---

## 🗂️ Project Structure

```
CSIT5970_Project/
├── main.py                      # CLI entry point (REPL / single Q / benchmark / index)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment variable template
│
├── config/
│   ├── agent_config.yaml        # Central configuration (LLM, RAG, tools, benchmark)
│   └── prompts/
│       ├── general.txt          # General-purpose persona
│       ├── linux_expert.txt     # Senior Linux/DevOps expert persona
│       └── security_expert.txt  # Cybersecurity expert persona
│
├── agent/
│   ├── core.py                  # Agent class — ReAct loop, LLM dispatch
│   ├── persona.py               # PersonaManager — load / cache / register prompts
│   └── memory.py                # MemoryManager — short-term buffer + RAG delegation
│
├── tools/
│   ├── base.py                  # BaseTool ABC + ToolRegistry
│   ├── knowledge_base.py        # RAG retrieval tool
│   ├── code_executor.py         # Sandboxed Python / Bash execution
│   └── web_search.py            # DuckDuckGo search (no API key required)
│
├── rag/
│   ├── vectorstore.py           # ChromaDB wrapper (add / query / count)
│   ├── indexer.py               # Document loader + chunker (txt/pdf/docx/md)
│   └── retriever.py             # High-level RAGRetriever returning Document objects
│
├── benchmark/
│   ├── evaluator.py             # BenchmarkEvaluator — 3-mode runner + LLM judge
│   ├── metrics.py               # EvaluationMetrics (mean, std, pass_rate, …)
│   └── datasets/
│       └── qa_test_set.json     # 25 expert Q&A pairs across 6 categories
│
├── data/
│   └── knowledge_base/
│       ├── linux_sysadmin.md    # Kernel tuning, log rotation, performance commands
│       └── security_devops.md   # OWASP Top 10, K8s security, DB migration
│
└── tests/
    ├── test_agent.py            # Agent core + persona + memory unit tests
    ├── test_tools.py            # ToolRegistry + CodeExecutor + WebSearch tests
    └── test_rag.py              # EvaluationMetrics + (optional) ChromaDB tests
```

---

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/Renegade-3863/CSIT5970_Project.git
cd CSIT5970_Project

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Index the knowledge base

```bash
python main.py --index-kb
```

### 4. Run the agent

```bash
# Interactive REPL
python main.py

# Single question with linux_expert persona
python main.py --persona linux_expert --question "How do I tune Linux kernel for PostgreSQL?"

# Run full benchmark evaluation
python main.py --benchmark
```

---

## 🐳 Docker Deployment

### Build and run

```bash
# Build the image
docker compose build

# Start the interactive agent
docker compose run --rm agent

# Index knowledge base
docker compose --profile indexer run --rm kb_indexer

# Run benchmark
docker compose --profile benchmark run --rm benchmark
```

### Cloud deployment (AWS example)

```bash
# On an AWS EC2 instance (t3.medium or larger)
git clone https://github.com/Renegade-3863/CSIT5970_Project.git
cd CSIT5970_Project
cp .env.example .env && nano .env   # Add API keys

docker compose build
docker compose --profile indexer run --rm kb_indexer  # Index once
docker compose run --rm agent                         # Interactive session
```

---

## 🧠 Personas

Three expert personas are available via the `--persona` flag or `config/agent_config.yaml`:

| Persona | Expertise |
|---|---|
| `general` | General-purpose assistant |
| `linux_expert` | Senior Linux/DevOps/SRE engineer — kernel, containers, CI/CD, cloud |
| `security_expert` | Cybersecurity professional — pen testing, OWASP, AD attacks, forensics |

### Adding a custom persona

1. Create `config/prompts/my_persona.txt` with the system prompt text.
2. Use it: `python main.py --persona my_persona`

---

## 📚 RAG Knowledge Base

The RAG system uses **ChromaDB** with **all-MiniLM-L6-v2** sentence-transformer embeddings (runs locally, no API key needed).

### Indexing custom documents

Place `.txt`, `.md`, `.pdf`, or `.docx` files in `data/knowledge_base/` and run:

```bash
python main.py --index-kb
```

### Configuration (`config/agent_config.yaml`)

```yaml
rag:
  enabled: true
  collection_name: knowledge_base
  persist_directory: ./data/chroma_db
  embedding_model: all-MiniLM-L6-v2
  top_k: 5
  similarity_threshold: 0.4
```

---

## 📊 Benchmark Evaluation

The benchmark evaluates three agent configurations using **LLM-as-a-Judge** (GPT-4o):

| Mode | Description |
|---|---|
| `llm_only` | Direct LLM call, no tools, no persona |
| `vanilla_agent` | ReAct agent with tools, default persona, **no RAG** |
| `custom_agent_with_rag` | Full agent: expert persona + RAG + all tools |

### Test set

`benchmark/datasets/qa_test_set.json` contains **25 questions** across:
- Linux system administration (kernel tuning, networking, disk management)
- Security (SQL injection, Kerberoasting, buffer overflow)
- Cloud & DevOps (AWS architecture, CI/CD, Kubernetes)
- RAG-specific questions (require knowledge base retrieval)

### Running the benchmark

```bash
python main.py --benchmark
```

Results are saved to `benchmark/results/` as JSON reports.

### Expected results pattern

```
============================================================
BENCHMARK SUMMARY
============================================================
Mode                           Avg Score  Avg Latency
------------------------------------------------------------
custom_agent_with_rag               8.4        12.3s
vanilla_agent                       6.7         8.1s
llm_only                            5.9         2.4s
============================================================
```

---

## 🧪 Tests

```bash
# Run all fast unit tests (no API keys or ChromaDB required)
pytest tests/ -v

# Include ChromaDB integration tests (requires chromadb installed)
RUN_CHROMA_TESTS=1 pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

---

## ⚙️ Configuration Reference

All settings live in `config/agent_config.yaml`:

```yaml
agent:
  max_iterations: 10    # Max ReAct loop iterations
  verbose: true         # Log thoughts and observations

llm:
  provider: openai      # openai | anthropic
  model: gpt-4o-mini    # Model name
  temperature: 0.1      # Lower = more deterministic
  max_tokens: 4096

persona:
  default: linux_expert # Default persona name

rag:
  enabled: true
  top_k: 5              # Documents to retrieve per query
  similarity_threshold: 0.4  # Minimum cosine similarity

tools:
  enabled:
    - knowledge_base
    - code_executor
    - web_search

benchmark:
  judge_model: gpt-4o   # Model used as evaluator
  modes:
    - llm_only
    - vanilla_agent
    - custom_agent_with_rag
```

---

## 📖 References

- [OpenHands (OpenDevin)](https://github.com/All-Hands-AI/OpenHands) — Software engineering agent
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) — General-purpose autonomous agent
- [LangGraph](https://github.com/langchain-ai/langgraph) — Multi-agent orchestration
- [CrewAI](https://github.com/crewAIInc/crewAI) — Role-based multi-agent framework
- [ChromaDB](https://github.com/chroma-core/chroma) — Open-source vector database
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework
- [SWE-bench](https://swebench.com) — Software engineering benchmark
- [AgentBench](https://github.com/THUDM/AgentBench) — Agent capability evaluation

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.