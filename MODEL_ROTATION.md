# 🔀 Multi-Model Rotation System

The Council of Infinite Innovators uses a **daily model rotation system** to leverage your HuggingFace Pro account access to multiple premium models.

## How It Works

Each of the 10 agent archetypes has **3 preferred models** that rotate daily based on the day of the year. This ensures:
- **Model diversity**: Different agents use different models simultaneously
- **Daily variety**: Same agent uses different models on different days
- **Performance testing**: Easy comparison of which models work best for which tasks
- **Cost optimization**: Automatic load distribution across models

## Today's Assignments (Day 311)

```
Rotation Index: 2 (day 311 % 3 = 2)
Each agent uses their 3rd preferred model:

strategist           → Llama-3.1-8B-Instruct       (business strategy)
architect            → Mistral-7B-Instruct-v0.3     (system design)
engineer             → Phi-3.5-mini-instruct        (code implementation)
designer             → Mistral-7B-Instruct-v0.3     (UX/UI design)
entrepreneur         → Llama-3.1-8B-Instruct       (market validation)
futurist             → Qwen2.5-7B-Instruct         (trend analysis)
economist            → Llama-3.2-3B-Instruct       (financial modeling)
ethicist             → Mistral-7B-Instruct-v0.3     (AI ethics)
philosopher          → gemma-2-9b-it               (deep thinking)
cultural_translator  → gemma-2-9b-it               (localization)
```

## Rotation Schedule

### Day 309 (Index 0)
- Strategist → Mistral-7B
- Engineer → Llama-3.2-3B
- Philosopher → Qwen2.5-7B
- ...

### Day 310 (Index 1)
- Strategist → Qwen2.5-7B
- Engineer → Qwen2.5-7B
- Philosopher → Phi-3.5
- ...

### Day 311 (Index 2) - TODAY
- Strategist → Llama-3.1-8B
- Engineer → Phi-3.5
- Philosopher → Gemma-2-9B
- ...

## Available Models (12 Total)

### Meta Llama Family
- `meta-llama/Llama-3.2-3B-Instruct` - Fast, efficient general purpose
- `meta-llama/Llama-3.2-1B-Instruct` - Ultra-fast for simple tasks
- `meta-llama/Llama-3.1-8B-Instruct` - Powerful reasoning and analysis

### Mistral AI
- `mistralai/Mistral-7B-Instruct-v0.3` - Excellent instruction following
- `mistralai/Mistral-Nemo-Instruct-2407` - Latest Mistral architecture

### Alibaba Qwen
- `Qwen/Qwen2.5-7B-Instruct` - Strong multilingual capabilities
- `Qwen/Qwen2.5-3B-Instruct` - Fast multilingual inference

### Google Gemma
- `google/gemma-2-9b-it` - Advanced reasoning
- `google/gemma-2-2b-it` - Lightweight alternative

### Microsoft Phi
- `microsoft/Phi-3.5-mini-instruct` - Optimized for code and logic

### Others
- `HuggingFaceH4/zephyr-7b-beta` - Community fine-tuned
- `01-ai/Yi-6B-Chat` - Chinese-English bilingual

## Agent Model Preferences

Each agent has specialized model preferences based on their role:

### Strategist (Business Strategy)
1. Mistral-7B - Excellent for structured analysis
2. Qwen2.5-7B - Strong business reasoning
3. Llama-3.1-8B - Deep strategic thinking

### Engineer (Code Implementation)
1. Llama-3.2-3B - Fast code generation
2. Qwen2.5-7B - Good at technical tasks
3. Phi-3.5 - Optimized for coding

### Designer (UX/UI)
1. Gemma-2-9B - Creative thinking
2. Llama-3.2-3B - Clear explanations
3. Mistral-7B - Structured design thinking

### Philosopher (Deep Analysis)
1. Qwen2.5-7B - Complex reasoning
2. Phi-3.5 - Logical analysis
3. Gemma-2-9B - Abstract thinking

*Full preferences defined in `council/model_rotation.py`*

## CLI Commands

### View today's assignments
```bash
python -m cli.app models
```

### Run single agent (uses assigned model)
```bash
python -m cli.app run --agent strategist --input "Your question"
# Today: Uses Llama-3.1-8B automatically
```

### Run multi-agent council (each uses different model)
```bash
python -m cli.app council --agents strategist,engineer,designer --input "Design an app"
# strategist → Llama-3.1-8B
# engineer   → Phi-3.5
# designer   → Mistral-7B
```

### Discover models from your HF account
```bash
python scripts/discover_hf_models.py
python scripts/discover_hf_models.py --test  # Test inference availability
```

## Your HuggingFace Models

Found 1 model in your account (`aliAIML`):
- `aliAIML/UI-TARS-1.5-7B` - Not compatible with chat API (general model)

**Note**: Your model is not a chat model, so it can't be used with the Inference API's chat endpoint. The rotation system uses public HF Pro models instead.

## Adding Custom Models

### From your HF account
```python
from council.model_rotation import add_custom_model

# Add your fine-tuned model
add_custom_model("my_model", "aliAIML/my-chat-model")
```

### Update agent preferences
Edit `council/model_rotation.py`:
```python
AGENT_MODEL_PREFERENCES = {
    "strategist": ["my_model", "mistral-7b", "qwen-2.5-7b"],
    # ...
}
```

## Benefits

### 1. **Model Comparison**
Run the same query on different days to compare:
- Day 309: Strategist uses Mistral-7B
- Day 310: Strategist uses Qwen2.5-7B
- Day 311: Strategist uses Llama-3.1-8B

### 2. **Cost Efficiency**
- Spread load across 12 different models
- Avoid rate limits on single model
- Leverage HF Pro quotas effectively

### 3. **Performance Optimization**
- Track which models perform best for which tasks
- Adjust preferences based on results
- Easy A/B testing across models

### 4. **Automatic Failover**
If one model is down or slow, others continue working

## Monitoring

Add logging to track model performance:

```python
# In council/llm.py
import time
start = time.time()
response = await llm.chat_completion(...)
duration = time.time() - start

logger.info(f"Model: {model_path}, Duration: {duration:.2f}s, Tokens: {len(response)}")
```

## Future Enhancements

1. **Performance tracking database**
   - Log model, agent, task, response time, quality score
   - Auto-optimize model preferences based on metrics

2. **Load-based rotation**
   - Check model availability before assignment
   - Use fastest available model dynamically

3. **Task-specific models**
   - Code tasks → Phi-3.5, DeepSeek-Coder
   - Creative tasks → Gemma-2, Yi-Chat
   - Analysis tasks → Llama-3.1, Qwen2.5

4. **Your custom models**
   - Fine-tune models for specific agents
   - Upload to HF and auto-add to rotation

## Environment Variables

```bash
# Required
HF_API_TOKEN=hf_your_token_here  # HF Pro for premium models

# Optional overrides
HF_INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct  # Default fallback
DEFAULT_PROVIDER=hf_inference
```

## Testing

```bash
# Test rotation module
python -m council.model_rotation

# Test single agent
python -m cli.app run --agent strategist --input "Test query"

# Test multi-agent with different models
python -m cli.app council --agents strategist,engineer --input "Test query"

# Run full test suite
pytest tests/ -v
```

## Troubleshooting

### Model not found
- Ensure HF_API_TOKEN is set
- Check model is available with HF Pro
- Verify model supports chat completion API

### Rate limiting
- Rotation helps avoid this
- Upgrade to HF Enterprise for higher quotas
- Add retry logic with exponential backoff

### Slow responses
- Some models (9B, 8B) are slower
- Use smaller models (1B, 2B, 3B) for speed
- Check `python -m cli.app models` for current assignments

---

**Powered by your HuggingFace Pro account** 🚀

Daily rotation ensures every agent leverages the best models available while automatically distributing load and enabling performance comparison.
