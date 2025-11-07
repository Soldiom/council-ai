"""
🚨 COLAB EMERGENCY FIX - Run This Cell NOW!

This will check your Colab status and restart the continuous learning system.
"""

import os
import json
from datetime import datetime

print("="*60)
print("🔍 COLAB DIAGNOSTICS - Checking Status...")
print("="*60)
print()

# 1. Check if we're in Colab
try:
    import google.colab
    print("✅ Running in Google Colab")
    IN_COLAB = True
except:
    print("❌ NOT in Google Colab!")
    IN_COLAB = False

print()

# 2. Check repository
repo_path = "/content/council-ai"
if os.path.exists(repo_path):
    print(f"✅ Repository exists at: {repo_path}")
    os.chdir(repo_path)
else:
    print(f"❌ Repository NOT found at: {repo_path}")
    print("🔧 Cloning repository...")
    os.system("git clone https://github.com/Soldiom/council-ai.git /content/council-ai")
    os.chdir(repo_path)

print()

# 3. Check training data
data_file = "training_data/agi_audit_log.jsonl"
if os.path.exists(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        count = len(lines)
    
    print(f"✅ Training data file exists")
    print(f"📊 Total examples: {count}")
    print(f"⏰ Estimated time: {(count // 50) * 30} minutes")
    print(f"🎯 Target for training: 600 examples")
    
    if count > 0:
        # Show last entry
        last_entry = json.loads(lines[-1])
        print(f"📅 Last update: {last_entry.get('timestamp', 'Unknown')}")
    
    if count < 50:
        print("⚠️ Very few examples - system just started or restarted")
    elif count >= 600:
        print("🎉 READY TO TRAIN! You have enough data!")
else:
    print(f"❌ No training data yet")
    print("🔧 Creating directory...")
    os.makedirs("training_data", exist_ok=True)
    count = 0

print()

# 4. Check if engine is running
print("="*60)
print("🔧 RESTARTING CONTINUOUS LEARNING ENGINE...")
print("="*60)
print()

# Import required modules
import sys
sys.path.append('/content/council-ai')

from council.continuous_learning import ContinuousLearningEngine

# Initialize engine
engine = ContinuousLearningEngine(
    hf_token=os.environ.get("HF_TOKEN"),
    collection_interval_minutes=30,
    training_interval_hours=6
)

print("✅ Engine initialized!")
print()
print("🚀 Starting continuous learning...")
print("📊 Current data count:", count)
print("🔄 Will collect 50 examples every 30 minutes")
print("🎓 Will train models after 600 examples (~6 hours)")
print()
print("="*60)
print("⚠️ KEEP THIS CELL RUNNING!")
print("⚠️ Do NOT close this browser tab!")
print("⚠️ Check progress every 1-2 hours")
print("="*60)
print()

# Start the engine
engine.start_continuous_learning()
