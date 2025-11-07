"""
🔥 REAL LEARNING FROM HUGGINGFACE & GITHUB
Collects REAL data from:
- HuggingFace datasets (AI conversations, code, documentation)
- GitHub repositories (code, issues, discussions)
- Public APIs and knowledge bases
"""

print("🚀 COUNCIL AI - REAL LEARNING SYSTEM")
print("=" * 70)
print("📚 Learning from: HuggingFace + GitHub + Real AI datasets")
print()

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    "transformers",
    "datasets", 
    "huggingface-hub",
    "requests",
    "beautifulsoup4",
    "PyGithub"
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✅ Dependencies installed!")
print()

# ============================================================================
# STEP 2: SETUP
# ============================================================================

print("📥 Setting up...")
import os
import json
import random
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import HfApi
import requests

# Clone repository if needed
if not os.path.exists('/content/council-ai'):
    subprocess.check_call(["git", "clone", "https://github.com/Soldiom/council-ai.git", "/content/council-ai"])
    print("✅ Repository cloned!")
else:
    print("✅ Repository exists!")

os.makedirs('/content/council-ai/training_data', exist_ok=True)
data_file = '/content/council-ai/training_data/agi_audit_log.jsonl'

print()

# ============================================================================
# STEP 3: REAL DATA SOURCES
# ============================================================================

class RealDataCollector:
    """Collects real training data from multiple sources"""
    
    def __init__(self):
        self.hf_api = HfApi()
        self.examples_collected = 0
        
    def collect_from_huggingface_datasets(self, num_examples=20):
        """Learn from HuggingFace public datasets"""
        print("📚 Learning from HuggingFace datasets...")
        examples = []
        
        try:
            # Dataset 1: OpenAssistant conversations (real AI chat data)
            print("  → Loading OpenAssistant conversations...")
            dataset = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
            
            count = 0
            for item in dataset:
                if count >= num_examples // 3:
                    break
                    
                # Get conversation pairs
                if item.get('text') and len(item['text']) > 10:
                    examples.append({
                        "timestamp": datetime.now().isoformat(),
                        "input": item['text'][:500],  # First 500 chars as input
                        "output": item['text'][-500:] if len(item['text']) > 500 else item['text'],
                        "metadata": {
                            "source": "huggingface_oasst1",
                            "agent": "unified",
                            "quality": "high"
                        }
                    })
                    count += 1
                    
            print(f"  ✅ Collected {count} examples from OpenAssistant")
            
        except Exception as e:
            print(f"  ⚠️  OpenAssistant error: {e}")
        
        try:
            # Dataset 2: Code instructions (programming knowledge)
            print("  → Loading code instruction dataset...")
            dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
            
            count = 0
            for item in dataset:
                if count >= num_examples // 3:
                    break
                    
                if item.get('instruction') and item.get('output'):
                    examples.append({
                        "timestamp": datetime.now().isoformat(),
                        "input": item['instruction'],
                        "output": item['output'],
                        "metadata": {
                            "source": "huggingface_code_instructions",
                            "agent": "code_assistant",
                            "quality": "high"
                        }
                    })
                    count += 1
                    
            print(f"  ✅ Collected {count} examples from code instructions")
            
        except Exception as e:
            print(f"  ⚠️  Code instructions error: {e}")
        
        try:
            # Dataset 3: Wikipedia snippets (general knowledge)
            print("  → Loading Wikipedia knowledge...")
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            
            count = 0
            for item in dataset:
                if count >= num_examples // 3:
                    break
                    
                if item.get('text') and len(item['text']) > 200:
                    # Create Q&A from Wikipedia content
                    text = item['text']
                    title = item.get('title', 'Unknown')
                    
                    examples.append({
                        "timestamp": datetime.now().isoformat(),
                        "input": f"Tell me about {title}",
                        "output": text[:800],  # First 800 chars
                        "metadata": {
                            "source": "huggingface_wikipedia",
                            "agent": "unified",
                            "topic": title,
                            "quality": "high"
                        }
                    })
                    count += 1
                    
            print(f"  ✅ Collected {count} examples from Wikipedia")
            
        except Exception as e:
            print(f"  ⚠️  Wikipedia error: {e}")
        
        return examples
    
    def collect_from_github(self, num_examples=15):
        """Learn from GitHub repositories"""
        print("📂 Learning from GitHub repositories...")
        examples = []
        
        try:
            # Popular AI/ML repositories to learn from
            repos = [
                "huggingface/transformers",
                "openai/openai-python",
                "langchain-ai/langchain",
                "pytorch/pytorch",
                "tensorflow/tensorflow",
            ]
            
            for repo_name in repos[:3]:  # Use first 3 repos
                try:
                    # Get README content (documentation)
                    url = f"https://api.github.com/repos/{repo_name}/readme"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        readme_data = response.json()
                        
                        # Decode base64 content
                        import base64
                        content = base64.b64decode(readme_data['content']).decode('utf-8')
                        
                        # Extract sections
                        sections = content.split('\n## ')
                        for section in sections[1:3]:  # Get 2 sections per repo
                            if len(section) > 100:
                                lines = section.split('\n')
                                title = lines[0]
                                body = '\n'.join(lines[1:])[:800]
                                
                                examples.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "input": f"Explain {title} in {repo_name}",
                                    "output": body,
                                    "metadata": {
                                        "source": f"github_{repo_name.replace('/', '_')}",
                                        "agent": "code_assistant",
                                        "type": "documentation",
                                        "quality": "high"
                                    }
                                })
                                
                                if len(examples) >= num_examples:
                                    break
                    
                    print(f"  ✅ Learned from {repo_name}")
                    
                except Exception as e:
                    print(f"  ⚠️  {repo_name} error: {e}")
                    continue
                    
                if len(examples) >= num_examples:
                    break
                    
        except Exception as e:
            print(f"  ⚠️  GitHub error: {e}")
        
        return examples
    
    def collect_from_arxiv_abstracts(self, num_examples=15):
        """Learn from arXiv research papers"""
        print("📄 Learning from arXiv research papers...")
        examples = []
        
        try:
            # Use arXiv API to get recent AI papers
            topics = ["artificial+intelligence", "machine+learning", "deep+learning", "neural+networks"]
            
            for topic in topics[:2]:  # Use first 2 topics
                try:
                    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=5"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        # Parse XML response
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(response.content)
                        
                        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                            title = entry.find('{http://www.w3.org/2005/Atom}title').text
                            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                            
                            examples.append({
                                "timestamp": datetime.now().isoformat(),
                                "input": f"What is the research about: {title}?",
                                "output": summary.strip(),
                                "metadata": {
                                    "source": "arxiv",
                                    "agent": "unified",
                                    "type": "research",
                                    "topic": topic.replace('+', ' '),
                                    "quality": "high"
                                }
                            })
                            
                            if len(examples) >= num_examples:
                                break
                        
                        print(f"  ✅ Learned from {topic.replace('+', ' ')} papers")
                        
                except Exception as e:
                    print(f"  ⚠️  {topic} error: {e}")
                    continue
                    
                if len(examples) >= num_examples:
                    break
                    
        except Exception as e:
            print(f"  ⚠️  arXiv error: {e}")
        
        return examples
    
    def collect_real_data(self, total_examples=50):
        """Collect from all sources"""
        print(f"\n🔥 COLLECTING {total_examples} REAL EXAMPLES...")
        print("=" * 70)
        print()
        
        all_examples = []
        
        # Distribute across sources
        try:
            # 40% from HuggingFace datasets
            hf_examples = self.collect_from_huggingface_datasets(int(total_examples * 0.4))
            all_examples.extend(hf_examples)
            print()
            
            # 30% from GitHub
            github_examples = self.collect_from_github(int(total_examples * 0.3))
            all_examples.extend(github_examples)
            print()
            
            # 30% from arXiv
            arxiv_examples = self.collect_from_arxiv_abstracts(int(total_examples * 0.3))
            all_examples.extend(arxiv_examples)
            print()
            
        except Exception as e:
            print(f"⚠️  Collection error: {e}")
        
        # Shuffle for diversity
        random.shuffle(all_examples)
        
        return all_examples[:total_examples]

# ============================================================================
# STEP 4: COLLECT REAL DATA NOW!
# ============================================================================

collector = RealDataCollector()
examples = collector.collect_real_data(total_examples=50)

print("=" * 70)
print(f"📊 Collected {len(examples)} REAL examples!")
print()

# Save to file
print("💾 Saving to training data...")
with open(data_file, 'a', encoding='utf-8') as f:
    for example in examples:
        f.write(json.dumps(example) + '\n')

print(f"✅ Saved to: {data_file}")
print(f"📈 File size: {os.path.getsize(data_file) / 1024:.1f} KB")
print()

# Show statistics
sources = {}
for ex in examples:
    source = ex['metadata'].get('source', 'unknown')
    sources[source] = sources.get(source, 0) + 1

print("📊 Data sources breakdown:")
for source, count in sources.items():
    print(f"  • {source}: {count} examples")
print()

print("=" * 70)
print("🎉 REAL DATA COLLECTION COMPLETE!")
print("✅ Your AI is learning from:")
print("   • HuggingFace datasets (OpenAssistant, Code, Wikipedia)")
print("   • GitHub repositories (documentation, code)")
print("   • arXiv research papers")
print()

# ============================================================================
# STEP 5: CONTINUOUS REAL LEARNING (every 30 minutes)
# ============================================================================

print("🔄 STARTING CONTINUOUS REAL LEARNING...")
print("💡 Will collect 50 more REAL examples every 30 minutes")
print("⏰ Next collection in 30 minutes...")
print()

import time
cycle = 2

while True:
    try:
        # Wait 30 minutes
        time.sleep(1800)
        
        print(f"\n🔥 CYCLE #{cycle} - COLLECTING REAL DATA...")
        print("=" * 70)
        print()
        
        # Collect more real data
        new_examples = collector.collect_real_data(total_examples=50)
        
        # Save
        with open(data_file, 'a', encoding='utf-8') as f:
            for example in new_examples:
                f.write(json.dumps(example) + '\n')
        
        # Count total
        with open(data_file, 'r') as f:
            total = len(f.readlines())
        
        print()
        print(f"🎉 Cycle #{cycle} complete!")
        print(f"📊 Total examples: {total}")
        print(f"📈 Progress to training: {(total / 600) * 100:.1f}%")
        print()
        
        if total >= 600:
            print("=" * 70)
            print("🔥 600+ EXAMPLES! READY TO TRAIN!")
            print("=" * 70)
            print()
        
        cycle += 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Stopped at cycle {cycle - 1}")
        break
    except Exception as e:
        print(f"⚠️  Error in cycle #{cycle}: {e}")
        print("🔄 Retrying in 5 minutes...")
        time.sleep(300)  # Wait 5 min before retry
        cycle += 1

print()
print("=" * 70)
print("✅ REAL LEARNING COMPLETE!")
print("=" * 70)
