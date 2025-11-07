"""
Test Deepfake & Document Forgery Detection
===========================================

Quick test of the new detection capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_deepfake_detection():
    """Test deepfake detection."""
    print("\n" + "=" * 70)
    print("🎭 TESTING DEEPFAKE DETECTION")
    print("=" * 70 + "\n")
    
    from council.agents.deepfake_detector import get_deepfake_detector
    
    detector = get_deepfake_detector()
    
    # Test cases
    test_cases = [
        {
            "media": "Video showing person's face with unnatural eye movements and blurry edges around face boundary. Lighting on face doesn't match background. Possible lip-sync mismatch detected.",
            "media_type": "video",
            "description": "Suspicious deepfake video"
        },
        {
            "media": "High-quality photo of person with overly smooth skin texture, symmetric facial features, and dead/lifeless eyes. Background has normal noise but face appears perfectly rendered.",
            "media_type": "image",
            "description": "AI-generated portrait"
        },
        {
            "media": "Audio recording of speech with robotic qualities, unnatural pauses, and spectral anomalies. Breathing patterns are inconsistent. Background noise suddenly changes.",
            "media_type": "audio",
            "description": "Cloned voice"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Media Type: {test['media_type']}")
        print(f"   Input: {test['media'][:80]}...")
        
        result = await detector.analyze(
            test["media"],
            {"media_type": test["media_type"]}
        )
        
        print(f"\n   🎯 Verdict: {result['verdict']}")
        print(f"   📊 Authenticity: {result['authenticity_score']:.0%}")
        print(f"   ⚠️  Indicators: {sum(len(v) for v in result['manipulation_indicators'].values())}")
        
        if result['manipulation_indicators']['ai_signatures']:
            print(f"   🤖 AI Signatures: {', '.join(result['manipulation_indicators']['ai_signatures'][:3])}")
    
    print("\n✅ Deepfake detection test complete!\n")


async def test_document_forgery():
    """Test document forgery detection."""
    print("=" * 70)
    print("📄 TESTING DOCUMENT FORGERY DETECTION")
    print("=" * 70 + "\n")
    
    from council.agents.document_forgery_detector import get_document_forgery_detector
    
    detector = get_document_forgery_detector()
    
    # Test cases
    test_cases = [
        {
            "document": "Passport scan showing missing holographic overlay, incorrect font on MRZ line, ghost image is absent. Issue date is 2024 but document shows heavy wear patterns. Photo has cut marks around edges.",
            "doc_type": "passport",
            "description": "Suspicious passport"
        },
        {
            "document": "National ID card with misaligned text fields, color mismatch between photo laminate and card body. UV-reactive ink is missing. Microprinting appears pixelated under magnification.",
            "doc_type": "id_card",
            "description": "Potentially forged ID"
        },
        {
            "document": "Driver's license with correct hologram, proper UV features, clean lamination, consistent fonts and alignment. Issue/expiry dates follow standard format. No visible tampering.",
            "doc_type": "drivers_license",
            "description": "Genuine driver's license"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['description']}")
        print(f"   Document Type: {test['doc_type']}")
        print(f"   Input: {test['document'][:80]}...")
        
        result = await detector.analyze(
            test["document"],
            {"document_type": test["doc_type"]}
        )
        
        print(f"\n   🎯 Verdict: {result['verdict']}")
        print(f"   📊 Authenticity: {result['authenticity_score']:.0%}")
        print(f"   ⚠️  Indicators: {sum(len(v) for v in result['forgery_indicators'].values())}")
        
        if result['forgery_indicators']['missing_security_features']:
            print(f"   🔒 Missing Security: {', '.join(result['forgery_indicators']['missing_security_features'][:3])}")
        
        if result['extracted_data']:
            print(f"   📋 Extracted: {result['extracted_data']}")
    
    print("\n✅ Document forgery detection test complete!\n")


async def main():
    """Run all tests."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   TESTING NEW DETECTION CAPABILITIES                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    try:
        # Test deepfake detection
        await test_deepfake_detection()
        
        # Test document forgery detection
        await test_document_forgery()
        
        print("=" * 70)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 70)
        print("\n📊 Training data saved in:")
        print("   • training_data/deepfake/")
        print("   • training_data/document_forgery/")
        print("\n💡 To build models with this data:")
        print("   python scripts/build_unified_model.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
