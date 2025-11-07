# 🎛️ VS Code Agent Template — Council of Infinite Innovators

A production‑ready Visual Studio Code setup to build AI solutions backed by the **Council of Infinite Innovators** archetypes (Architect, Engineer, Strategist, etc.).

This template gives you:

* 🧠 **Multi‑agent runtime** (LangGraph‑style state machine, simple and robust)
* 🤖 **Agentic AI mode** (ReAct pattern with tool use: web search, code execution, calculations)
* 🌐 **Unified AI Platform** (Use ALL HuggingFace models, train YOUR unified model from user interactions!)
* 🔍 **Forensic AI Agent** (Specialized security analysis: logs, malware, threats → YOUR forensic model!)
*  **Model Ensemble** (Query GPT-4, Claude, Gemini, Llama simultaneously and combine outputs!)
* 🎓 **Continuous Learning** (Automatically collect training data and fine-tune YOUR custom models)
* 🔀 **Multi‑model rotation** (12+ premium models, daily rotation per agent)
* 🚀 **CLI + API** to run agents locally or serve over HTTP
* 🏗️ **Model Builder** (Builds YOUR models from ALL collected data - one command!)
* 🧰 **VS Code tasks** (run, test, lint, graphviz render)
* 🧪 **Testing harness** with deterministic "fixtures"
* 🔐 **.env secrets flow** (no secrets in code)
* 🧩 **Prompt packs** for each Council archetype
* 🧱 **Dev Container** for clean, reproducible envs
* 🧭 **Roadmap scaffolds** aligned to the 20‑Phase framework

---

## 0) Quick Start

**🚀 RECOMMENDED: Free Cloud Deployment (No downloads, no GPU needed!)**

Deploy to Railway, Render, or Fly.io with **FREE Hugging Face Inference API**:

```powershell
# 1) Get a free HF token: https://huggingface.co/settings/tokens
#    (Click "New token" → Read access is enough)

# 2) Deploy to Railway (easiest):
#    - Push this repo to GitHub
#    - Connect at railway.app
#    - Add HF_API_TOKEN secret in Railway dashboard
#    - Deploy! Railway uses railway.toml config automatically

# 3) Or deploy to Render:
#    - Connect GitHub repo at render.com
#    - Create web service from render.yaml
#    - Add HF_API_TOKEN in environment variables
#    - Deploy automatically

# 4) Or deploy to Fly.io:
flyctl launch  # Uses fly.toml config
flyctl secrets set HF_API_TOKEN=hf_your_token_here
flyctl deploy

# Your API will be live at: https://your-app.railway.app (or render/fly URL)
```

**💻 Local Development (Free API - No Downloads):**

```powershell
# 1) Get free HF token: https://huggingface.co/settings/tokens

# 2) Copy .env.example to .env
cp .env.example .env

# 3) Edit .env and add:
#    HF_API_TOKEN=hf_your_token_here
#    DEFAULT_PROVIDER=hf_inference

# 4) Install and run
pip install -r requirements.txt
python -m cli.app council --agents "strategist,architect" --input "Design a platform"

# 5) View today's model rotation
python -m cli.app models
# Shows which of the 12+ models each agent is using today

# 6) Or start API locally
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

**� Multi-Model Rotation** (with HF Pro):

Each agent automatically rotates through 3 preferred models daily:
- **12+ premium models**: Llama 3.2/3.1, Mistral 7B, Qwen 2.5, Gemma 2, Phi 3.5, and more
- **Daily rotation**: Different models for same agent each day (based on day of year % 3)
- **Agent specialization**: Engineer prefers code models, Strategist prefers reasoning models
- **View assignments**: `python -m cli.app models`

See **[MODEL_ROTATION.md](MODEL_ROTATION.md)** for details.

**Free Cloud Models** (via HF Inference API - recommended):
- `meta-llama/Llama-3.2-3B-Instruct` (fast, HF Pro)
- `meta-llama/Llama-3.1-8B-Instruct` (powerful reasoning, HF Pro)
- `mistralai/Mistral-7B-Instruct-v0.3` (excellent instruction following)
- `Qwen/Qwen2.5-7B-Instruct` (multilingual, strong reasoning)
- `google/gemma-2-9b-it` (advanced reasoning)
- `microsoft/Phi-3.5-mini-instruct` (optimized for code)

**💾 Local Models (Optional - Requires Download):**

```powershell
# If you want to run models offline on your machine:
# 1) Set DEFAULT_PROVIDER=huggingface in .env
# 2) Download a model
python scripts/download_model.py microsoft/phi-2
# 3) Run
python -m cli.app run --agent strategist --input "Your prompt"
```

**🤖 Agentic AI Mode** (NEW - with tool use and iterative reasoning):

```powershell
# Run agent in AGENTIC mode with tools (web search, code execution, calculations)
python -m cli.app agentic --agent strategist --input "Research AI market and calculate growth rate"

# Agent will:
# - Search web for current data
# - Perform calculations  
# - Iterate until task complete
# - Show reasoning trail
```

**Difference:**
- **Simple mode**: One prompt → one response (fast, basic)
- **Agentic mode**: Multi-step reasoning with tools (slower, powerful)

See **[AGENTIC_AI.md](AGENTIC_AI.md)** for full explanation of agentic AI vs simple prompt-based agents.

**🎭 Model Ensemble** (NEW - combine GPT-4, Claude, Gemini, Llama, Qwen):

```powershell
# Query multiple top models and combine their responses intelligently
python -m cli.app ensemble --input "Explain quantum computing" --models 3

# Models used: GPT-4, Claude Opus, Gemini Pro
# Strategy: Best-of-N (highest quality response)
# Result: Better than any single model!

# Check continuous learning progress
python -m cli.app learning-stats

# After collecting 100+ examples, fine-tune YOUR custom model

# OPTION A: OpenAI (quick but vendor lock-in)
python -m cli.app finetune --provider openai

# OPTION B: HuggingFace (YOU OWN THE MODEL!) ⭐ RECOMMENDED
python -m cli.app finetune --provider huggingface
# See QUICKSTART_OPTION_B.md for Google Colab guide (FREE GPU!)
```

**Benefits:**
- 🏆 Better quality (ensemble > single model)
- 🎓 Continuous learning (auto-collect training data)
- 💰 Cost optimization (95-100% cost reduction with YOUR model)
- 🔧 Full customization (model learns YOUR patterns)
- 🎯 Full ownership (Option B: YOU own the model, run offline, no vendor lock-in)

**Fine-Tuning Guides:**
- **Quick Start:** [QUICKSTART_OPTION_B.md](QUICKSTART_OPTION_B.md) - 3-step process
- **Google Colab (FREE GPU):** [COLAB_FINETUNING.md](COLAB_FINETUNING.md) - Step-by-step
- **Full Guide:** [OPTION_B_HUGGINGFACE.md](OPTION_B_HUGGINGFACE.md) - Complete details
- **Comparison:** [ENSEMBLE_AND_LEARNING.md](ENSEMBLE_AND_LEARNING.md) - All options

**🌐 Unified AI Platform** (NEW - Use ALL HuggingFace Models!):

```powershell
# Discover ALL HuggingFace models (100+ models across 25+ capabilities)
python -m cli.app unified --discover

# Start unified API
uvicorn api.unified:app --reload --port 8000

# Visit: http://localhost:8000/docs
# POST /task with any request → auto-routes to best model!

# Check platform stats
python -m cli.app unified --stats
```

**How it works:**
1. 🔍 **Discovers ALL HF models** (text, image, audio, translation, etc.)
2. 🎯 **Auto-routes** user requests to best model for task
3. 📊 **Collects training data** from ALL interactions
4. 🎓 **Trains YOUR unified model** daily on collected data
5. 🔄 **Auto-updates** with new HF models and improvements
6. 🚀 **Result:** ONE model that does EVERYTHING, gets smarter daily!

**Benefits:**
- ✅ Users interact with your platform → train YOUR model
- ✅ Support 25+ capabilities (translation, summarization, generation, etc.)
- ✅ ONE unified model (vs 100+ separate models)
- ✅ Improves daily from real usage
- ✅ 95% cost reduction after training
- ✅ Auto-discovers new HF models

See **[UNIFIED_PLATFORM.md](UNIFIED_PLATFORM.md)** for complete guide.

**🔍 Forensic AI Agent** (NEW - Security & Digital Forensics!):

```powershell
# Analyze security logs
python -m cli.app forensic --input "ERROR: Failed login from 192.168.1.100"

# Analyze malware
python -m cli.app forensic --input "Trojan detected: hash MD5:abc123..."

# Analyze network traffic
python -m cli.app forensic --input "Suspicious connection to 45.33.32.156:4444"

# Auto-extracts IOCs (IPs, hashes, CVEs, URLs)
# Assesses severity (Critical/High/Medium/Low)
# Saves as training data for YOUR forensic model!
```

**🏗️ Build YOUR Models** (NEW - One Command!):

```powershell
# Collects ALL training data from:
# - Ensemble interactions (GPT-4, Claude, Gemini)
# - Platform user data (unified API)
# - Forensic analysis (security logs, malware)

python -m cli.app build

# OR
python scripts/build_unified_model.py

# Output: Ready-to-train datasets for:
# 1. aliAIML/unified-ai-model (general purpose)
# 2. aliAIML/forensic-ai-model (security specialist)
```

**What you get:**
- 🌐 **Unified Model** - Handles everything (text, analysis, planning)
- 🔍 **Forensic Model** - Security expert (logs, malware, threats)
- 💰 **95% cost reduction** - Self-hosted, $0 inference
- 🎓 **Continuous learning** - Improves daily from usage
- 🎯 **Complete ownership** - YOUR models, YOUR data

See **[BUILD_NOW.md](BUILD_NOW.md)** for complete guide!

## 🎮 RTX 4060 + Google Colab Setup

**You have an RTX 4060?** Perfect for fine-tuning!

### **Quick Start (5 minutes):**

```powershell
# 1. Install CUDA PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 3. Run automated fine-tuning
python scripts/auto_local_gpu_finetune.py
```

**What you get:**
- ✅ **RTX 4060**: 2x faster than Colab FREE
- ✅ **Google Colab**: FREE backup when PC is busy
- ✅ **Automated**: Collects data + builds + trains
- ✅ **Cost**: $0.10/day (electricity) vs $50+/month (APIs)

**See:** [LOCAL_GPU_SETUP.md](LOCAL_GPU_SETUP.md) for complete guide

## 🤖 COMPLETE AUTOMATION - NO USER INPUT!

**Everything is automated:**

```powershell
# ONE COMMAND = EVERYTHING AUTOMATED
python scripts/auto_build_and_deploy.py
```

**What gets automated:**
- ✅ **Data Collection**: 35+ training examples across all domains
  - Ensemble queries (GPT-4, Claude, Gemini)
  - Forensic analysis (security logs, malware, threats)
  - Deepfake detection (video, image, audio manipulation)
  - Document forgery (passports, IDs, certificates)
- ✅ **Model Building**: Unified + Forensic + Deepfake + Document models
- ✅ **Dataset Preparation**: Ready-to-train .jsonl files
- ✅ **Google Colab Code**: Copy/paste and run (FREE GPU)
- ✅ **Daily Automation**: Schedule with Task Scheduler
- ✅ **Zero Configuration**: Uses existing API keys automatically

**Cost: $0** (100% free using Google Colab GPU)

See **[BUILD_NOW.md](BUILD_NOW.md)** for complete automation guide!

## Automation

Two automation helpers are included:

- GitHub Actions CI: `.github/workflows/ci.yml` — runs lint and tests on push/PR.
- Local runner script: `scripts/auto_run.py` — convenience wrapper to run setup, lint and tests or start the API.

Examples:

```powershell
python scripts/auto_run.py setup
python scripts/auto_run.py check
python scripts/auto_run.py api
python scripts/auto_run.py full
```

Note: `pip install -e .` may fail in the repository's flat layout; the script attempts it but does not fail the whole flow if editable install is not possible. Consider converting to a `src/` layout or explicitly listing packages in `pyproject.toml` for a reproducible editable install.
```

---

## Repository Layout

```
ai-council-agent/
├─ .devcontainer/
│  ├─ devcontainer.json
│  └─ Dockerfile
├─ .vscode/
│  ├─ extensions.json
│  ├─ launch.json
│  └─ tasks.json
├─ council/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ graph.py           # LangGraph-style state machine
│  ├─ memory.py          # simple vector memory + scratchpad
│  ├─ llm.py             # LLM adapters
│  ├─ tools/
│  │  ├─ web.py          # optional web search/citation tool
│  │  ├─ files.py        # local file read/write safe ops
│  │  └─ code.py         # code runner (sandboxed)
│  ├─ agents/
│  │  ├─ base.py
│  │  ├─ architect.py
│  │  ├─ entrepreneur.py
│  │  ├─ strategist.py
│  │  ├─ engineer.py
│  │  ├─ designer.py
│  │  ├─ futurist.py
│  │  ├─ economist.py
│  │  ├─ ethicist.py
│  │  ├─ philosopher.py
│  │  └─ cultural_translator.py
│  └─ prompts/
│     ├─ system/
│     │  ├─ meta.txt
│     │  └─ safety.txt
│     └─ archetypes/
│        ├─ architect.txt
│        ├─ entrepreneur.txt
│        ├─ strategist.txt
│        ├─ engineer.txt
│        ├─ designer.txt
│        ├─ futurist.txt
│        ├─ economist.txt
│        ├─ ethicist.txt
│        ├─ philosopher.txt
│        └─ cultural_translator.txt
├─ api/
│  ├─ main.py            # FastAPI server
│  └─ schemas.py
├─ cli/
│  └─ app.py             # Typer CLI
├─ tests/
│  ├─ test_agents.py
│  └─ fixtures/
│     └─ strategist_fixture.json
├─ plans/                # 20-Phase framework roadmaps
│  ├─ 01_audit.md
│  ├─ 04_feature_pipeline.md
│  ├─ 08_execution_roadmap.md
│  ├─ 14_resilience_security.md
│  └─ 17_data_analytics.md
├─ .env.example
├─ .gitignore
├─ Makefile
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## License

MIT