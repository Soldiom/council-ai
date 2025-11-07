"""
Example: Using YOUR Custom Fine-tuned Model

After fine-tuning, you'll have YOUR OWN model that combines
the knowledge of GPT-4, Claude, Gemini, and Llama!
"""

from huggingface_hub import InferenceClient
import os

# YOUR MODEL (after fine-tuning)
YOUR_MODEL_ID = "aliAIML/council-custom-model"  # ← This is YOURS!

# Initialize client
client = InferenceClient(
    model=YOUR_MODEL_ID,
    token=os.getenv("HF_API_TOKEN")
)

# Use YOUR model
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a strategic advisor."},
        {"role": "user", "content": "What are the top 3 AI trends?"}
    ],
    max_tokens=500,
)

print(response.choices[0].message.content)

# YOUR model knows:
# - What GPT-4 would say
# - What Claude would say  
# - What Gemini would say
# - What Llama would say
# But it's YOURS and costs 95% less!
