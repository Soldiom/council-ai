"""
Automated Training Data Collection
===================================

Automatically collects training data from ALL sources:
- Ensemble queries (GPT-4, Claude, Gemini)
- Forensic analysis samples
- Deepfake detection samples
- Document forgery detection samples

Runs continuously, collecting diverse examples.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import random

# Fix Windows encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).parent.parent))

from council.model_ensemble import get_ensemble, EnsembleStrategy
from council.agents.forensic import get_forensic_agent
from council.agents.deepfake_detector import get_deepfake_detector
from council.agents.document_forgery_detector import get_document_forgery_detector


# Sample queries for different domains
ENSEMBLE_QUERIES = [
    "Create a comprehensive cybersecurity incident response plan",
    "Design a microservices architecture for an e-commerce platform",
    "Explain quantum computing and its business applications",
    "Develop a machine learning model deployment strategy",
    "Write a market analysis for AI-powered customer service tools",
    "Create a blockchain-based supply chain solution",
    "Design a real-time fraud detection system",
    "Develop a content recommendation engine architecture",
    "Create a scalable data pipeline for IoT devices",
    "Design a multi-cloud disaster recovery strategy",
    "Develop an AI-powered code review system",
    "Create a zero-trust security architecture",
    "Design a serverless event-driven application",
    "Develop a customer churn prediction model",
    "Create a distributed caching strategy for high-traffic apps",
    "Design a CI/CD pipeline for containerized applications",
    "Develop a natural language processing chatbot",
    "Create a real-time analytics dashboard architecture",
    "Design a graph database schema for social networks",
    "Develop a predictive maintenance system using ML",
]

FORENSIC_SAMPLES = [
    {
        "input": "ERROR: Multiple failed SSH login attempts from 45.33.32.156 targeting root account. Failed attempts: 47 in 5 minutes. Source: China. User-Agent: automated scanner.",
        "type": "log_analysis"
    },
    {
        "input": "Malware detected: Trojan.Generic.KD.54321 MD5:d41d8cd98f00b204e9800998ecf8427e SHA256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 File: invoice.exe",
        "type": "malware_analysis"
    },
    {
        "input": "Suspicious outbound connection to 185.220.101.45:4444 detected. Protocol: TCP. Data transferred: 2.5 MB over 30 minutes. Process: svchost.exe (PID: 1337)",
        "type": "network_analysis"
    },
    {
        "input": "CVE-2024-1234: Remote code execution vulnerability in Apache Struts 2.5.30. CVSS Score: 9.8 (Critical). Exploit code publicly available on GitHub.",
        "type": "threat_intelligence"
    },
    {
        "input": "Ransomware activity: Files encrypted with .lockbit extension. Ransom note: README_TO_DECRYPT.txt. Bitcoin: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. Affected: 1,247 files.",
        "type": "malware_analysis"
    },
    {
        "input": "Port scan detected from 192.168.1.105 scanning ports 22, 23, 80, 443, 3389, 8080 on internal network. Scan rate: 100 ports/second. Tool: nmap",
        "type": "network_analysis"
    },
    {
        "input": "Privilege escalation attempt: User 'guest' tried to access admin panel at /wp-admin/. IP: 203.0.113.45. Failed authentication. Blocked by WAF.",
        "type": "log_analysis"
    },
    {
        "input": "SQL injection attempt detected: ' OR '1'='1 in login form. Source IP: 198.51.100.23. User-Agent: sqlmap/1.6. Request blocked.",
        "type": "log_analysis"
    },
    {
        "input": "Zero-day exploit targeting CVE-2024-9999 in Microsoft Exchange Server. Active exploitation in the wild. No patch available. Recommend disabling affected service.",
        "type": "threat_intelligence"
    },
    {
        "input": "DDoS attack detected: 50,000 requests/second from botnet. Attack type: SYN flood. Target: web server. Mitigation: Rate limiting + CloudFlare.",
        "type": "network_analysis"
    },
]

DEEPFAKE_SAMPLES = [
    {
        "media": "Video showing CEO announcing fake merger. Face has unnatural eye movements, blurry edges around face boundary. Lighting on face doesn't match background. Lip-sync appears slightly off.",
        "type": "video"
    },
    {
        "media": "Profile photo with overly smooth skin, perfectly symmetric features, lifeless eyes. Background noise normal but face perfectly rendered. Possible GAN-generated image.",
        "type": "image"
    },
    {
        "media": "Audio of politician making controversial statement. Voice has robotic qualities, unnatural pauses, spectral anomalies. Breathing patterns inconsistent.",
        "type": "audio"
    },
    {
        "media": "News anchor video with face swap artifacts. Facial expressions don't match voice emotion. Temporal inconsistencies between frames. Hair boundary shows warping.",
        "type": "video"
    },
    {
        "media": "Celebrity endorsement photo showing person holding product. Fingers appear distorted, lighting inconsistent, product logo has slight blur.",
        "type": "image"
    },
    {
        "media": "Phone call recording claiming to be from bank CEO. Voice cloning suspected - unnatural prosody, concatenation artifacts, spectral analysis shows synthesis patterns.",
        "type": "audio"
    },
    {
        "media": "Social media video of influencer. Checkerboard GAN artifacts visible on skin. Teeth unnaturally perfect. Background subtly warped near head boundaries.",
        "type": "video"
    },
    {
        "media": "ID photo submission with synthetic face. Eyes too symmetric, skin texture overly smooth, ears anatomically incorrect. AI detector confidence: 94% synthetic.",
        "type": "image"
    },
]

DOCUMENT_SAMPLES = [
    {
        "document": "Passport scan: Missing holographic overlay, incorrect font on MRZ line, no ghost image. Issue date 2024 but shows heavy wear. Photo has cut marks around edges.",
        "type": "passport"
    },
    {
        "document": "National ID: Misaligned text fields, color mismatch on laminate. UV-reactive ink missing. Microprinting pixelated under magnification. Document number format incorrect.",
        "type": "id_card"
    },
    {
        "document": "Driver's license: Correct hologram present, proper UV features, clean lamination, fonts aligned. Issue/expiry dates valid format. No tampering visible.",
        "type": "drivers_license"
    },
    {
        "document": "Passport: Correct security thread, valid watermarks, proper MRZ with correct check digits, hologram authentic. All security features present.",
        "type": "passport"
    },
    {
        "document": "Bank statement: Altered transaction amounts visible under magnification. Font inconsistency in modified rows. Digital PDF shows multiple save dates.",
        "type": "bank_statement"
    },
    {
        "document": "University diploma: Wrong paper type, missing embossed seal, signature appears scanned not original. Issuing institution name has typo.",
        "type": "certificate"
    },
    {
        "document": "ID card: Photo substitution evident - glue residue visible, photo quality doesn't match card print quality. Different paper layer underneath photo.",
        "type": "id_card"
    },
]


async def collect_ensemble_data(num_samples: int = 10):
    """Collect ensemble training data."""
    print(f"\n🎭 Collecting {num_samples} Ensemble Samples...")
    print("=" * 70)
    
    ensemble = get_ensemble()
    queries = random.sample(ENSEMBLE_QUERIES, min(num_samples, len(ENSEMBLE_QUERIES)))
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}/{num_samples}: {query[:60]}...")
        
        try:
            messages = [{"role": "user", "content": query}]
            result = await ensemble.ensemble_query(
                messages=messages,
                strategy=EnsembleStrategy.BEST_OF_N,
                num_models=2,
            )
            print(f"   ✅ Collected ({result.get('selected_model', 'N/A')})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Ensemble collection complete!")


async def collect_forensic_data(num_samples: int = 10):
    """Collect forensic training data."""
    print(f"\n🔍 Collecting {num_samples} Forensic Samples...")
    print("=" * 70)
    
    agent = get_forensic_agent()
    samples = random.sample(FORENSIC_SAMPLES, min(num_samples, len(FORENSIC_SAMPLES)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}/{num_samples}: {sample['type']} - {sample['input'][:50]}...")
        
        try:
            context = {"analysis_type": sample["type"]}
            result = await agent.analyze(sample["input"], context)
            print(f"   ✅ {result.get('severity', 'N/A')} severity")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Forensic collection complete!")


async def collect_deepfake_data(num_samples: int = 8):
    """Collect deepfake detection data."""
    print(f"\n🎭 Collecting {num_samples} Deepfake Samples...")
    print("=" * 70)
    
    detector = get_deepfake_detector()
    samples = random.sample(DEEPFAKE_SAMPLES, min(num_samples, len(DEEPFAKE_SAMPLES)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}/{num_samples}: {sample['type']} - {sample['media'][:50]}...")
        
        try:
            context = {"media_type": sample["type"]}
            result = await detector.analyze(sample["media"], context)
            print(f"   ✅ {result.get('verdict', 'N/A')} ({result.get('authenticity_score', 0):.0%})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Deepfake collection complete!")


async def collect_document_data(num_samples: int = 7):
    """Collect document forgery data."""
    print(f"\n📄 Collecting {num_samples} Document Samples...")
    print("=" * 70)
    
    detector = get_document_forgery_detector()
    samples = random.sample(DOCUMENT_SAMPLES, min(num_samples, len(DOCUMENT_SAMPLES)))
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}/{num_samples}: {sample['type']} - {sample['document'][:50]}...")
        
        try:
            context = {"document_type": sample["type"]}
            result = await detector.analyze(sample["document"], context)
            print(f"   ✅ {result.get('verdict', 'N/A')} ({result.get('authenticity_score', 0):.0%})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Document collection complete!")


async def main():
    """Automated data collection."""
    print("\n" + "=" * 70)
    print("        AUTOMATED TRAINING DATA COLLECTION")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Collect from all sources
        await collect_ensemble_data(10)
        await collect_forensic_data(10)
        await collect_deepfake_data(8)
        await collect_document_data(7)
        
        print("\n" + "=" * 70)
        print("*** AUTOMATED COLLECTION COMPLETE! ***")
        print("=" * 70)
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📊 Total samples collected: ~35")
        print("\n💡 Next: python scripts/build_unified_model.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
