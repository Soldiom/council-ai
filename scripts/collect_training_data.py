"""
Quick Training Data Collection
===============================

Collects training data quickly for unified + forensic models.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows encoding
import os
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).parent.parent))

from council.model_ensemble import get_ensemble, EnsembleStrategy
from council.agents.forensic import get_forensic_agent


async def collect_ensemble_samples():
    """Collect ensemble training data."""
    print("🎭 Collecting Ensemble Training Data...")
    print("=" * 70)
    
    ensemble = get_ensemble()
    
    # Sample queries across different domains
    queries = [
        "Create a comprehensive cybersecurity incident response plan",
        "Design a microservices architecture for an e-commerce platform",
        "Explain quantum computing and its business applications",
        "Develop a machine learning model deployment strategy",
        "Write a market analysis for AI-powered customer service tools",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Processing: {query[:50]}...")
        
        messages = [{"role": "user", "content": query}]
        
        result = await ensemble.ensemble_query(
            messages=messages,
            strategy=EnsembleStrategy.BEST_OF_N,
            num_models=2,  # Use 2 models for speed
        )
        
        print(f"   ✅ Collected ({result.get('selected_model', 'N/A')})")
    
    print(f"\n✅ Ensemble data collection complete!")


async def collect_forensic_samples():
    """Collect forensic training data."""
    print("\n🔍 Collecting Forensic Training Data...")
    print("=" * 70)
    
    agent = get_forensic_agent()
    
    # Sample forensic evidence
    samples = [
        {
            "input": "ERROR: Multiple failed SSH login attempts from 45.33.32.156 targeting root account. Failed attempts: 47 in 5 minutes.",
            "type": "log_analysis"
        },
        {
            "input": "Malware detected: Trojan.Generic.KD.54321 MD5:d41d8cd98f00b204e9800998ecf8427e SHA256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "type": "malware_analysis"
        },
        {
            "input": "Suspicious outbound connection to 185.220.101.45:4444 detected. Protocol: TCP. Data transferred: 2.5 MB over 30 minutes.",
            "type": "network_analysis"
        },
        {
            "input": "CVE-2024-1234: Remote code execution vulnerability in Apache Struts 2.5.30. CVSS Score: 9.8 (Critical). Exploit code publicly available.",
            "type": "threat_intelligence"
        },
        {
            "input": "Ransomware activity detected: Files encrypted with .lockbit extension. Ransom note found: README_TO_DECRYPT.txt. Bitcoin address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "type": "malware_analysis"
        },
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. Analyzing: {sample['input'][:50]}...")
        
        context = {"analysis_type": sample["type"]}
        result = await agent.analyze(sample["input"], context)
        
        print(f"   ✅ Collected ({sample['type']}) - Severity: {result.get('severity', 'N/A')}")
    
    print(f"\n✅ Forensic data collection complete!")


async def main():
    """Collect all training data."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║         QUICK TRAINING DATA COLLECTION                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    try:
        # Collect ensemble data
        await collect_ensemble_samples()
        
        # Collect forensic data
        await collect_forensic_samples()
        
        print("\n" + "=" * 70)
        print("✅ DATA COLLECTION COMPLETE!")
        print("=" * 70)
        print("\n📊 Next steps:")
        print("   1. Check data: python -m cli.app learning-stats")
        print("   2. Build models: python scripts/build_unified_model.py")
        print("   3. Fine-tune: Follow COLAB_FINETUNING.md")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
