"""
🧪 TEST ALL NEW FEATURES

Tests:
✅ Data Analytics
✅ Model Rotation
✅ Forensic Models
✅ Agentic AI
"""

import asyncio
from datetime import datetime


def test_analytics():
    """Test data analytics"""
    print("\n" + "=" * 80)
    print("🧪 TESTING DATA ANALYTICS")
    print("=" * 80)
    
    try:
        from council.data_analytics import get_analytics, DataCollectionMetrics
        
        analytics = get_analytics()
        
        # Simulate some data
        metrics = DataCollectionMetrics(
            timestamp=datetime.now().isoformat(),
            date=datetime.now().strftime('%Y-%m-%d'),
            hour=datetime.now().hour,
            ensemble_examples=50,
            forensic_examples=20,
            audio_examples=15,
            image_examples=10,
            models_used=["gpt-4o", "claude-3.5-sonnet", "whisper-large-v3"],
            models_count=3,
            avg_quality_score=8.5,
            estimated_cost=0.25
        )
        
        analytics.log_collection(metrics)
        analytics.print_daily_summary()
        
        print("✅ Analytics working!")
        return True
    except Exception as e:
        print(f"❌ Analytics failed: {e}")
        return False


def test_model_rotation():
    """Test model rotation system"""
    print("\n" + "=" * 80)
    print("🧪 TESTING MODEL ROTATION")
    print("=" * 80)
    
    try:
        from council.model_rotation import get_rotation_engine
        
        engine = get_rotation_engine(models_per_day=40)
        engine.print_daily_rotation()
        
        # Test getting model for task
        model = engine.get_model_for_task("image_generation")
        print(f"\n✅ Best model for image generation: {model.name}")
        
        return True
    except Exception as e:
        print(f"❌ Model rotation failed: {e}")
        return False


def test_forensic_models():
    """Test forensic models catalog"""
    print("\n" + "=" * 80)
    print("🧪 TESTING FORENSIC MODELS")
    print("=" * 80)
    
    try:
        from council.forensic_models import (
            print_forensic_catalog,
            get_best_model_for_task,
            ALL_FORENSIC_MODELS
        )
        
        print(f"\nTotal Forensic Models: {len(ALL_FORENSIC_MODELS)}")
        
        # Test getting best models
        tasks = [
            "transcribe_audio",
            "identify_speaker",
            "detect_image_deepfake",
            "verify_signature"
        ]
        
        print("\n📊 Best Models for Tasks:")
        for task in tasks:
            model = get_best_model_for_task(task)
            print(f"   {task}: {model.name} ({model.accuracy * 100:.1f}% accuracy)")
        
        print("\n✅ Forensic models working!")
        return True
    except Exception as e:
        print(f"❌ Forensic models failed: {e}")
        return False


async def test_agentic_ai():
    """Test agentic AI features"""
    print("\n" + "=" * 80)
    print("🧪 TESTING AGENTIC AI")
    print("=" * 80)
    
    try:
        from council.agi_features import (
            get_agentic_browser,
            get_human_like_ai,
            AGENTIC_MODELS
        )
        
        print(f"\nAgentic Models Available: {len(AGENTIC_MODELS)}")
        for name, model in AGENTIC_MODELS.items():
            print(f"   - {model.name}: Autonomy {model.autonomy_score}/10, Human-like {model.human_likeness}/10")
        
        # Test browser
        browser = get_agentic_browser()
        print(f"\n✅ Agentic browser ready: {browser.model.name}")
        
        # Test human-like AI
        human_ai = get_human_like_ai(personality="professional")
        response = human_ai.format_response(
            task="Test task",
            thinking_process=["Step 1: Analyze", "Step 2: Execute"],
            conclusion="Test complete",
            confidence=0.85
        )
        print(f"✅ Human-like AI ready: {len(response)} chars response")
        
        # Test autonomous research (quick demo)
        print("\n🤖 Testing autonomous research...")
        result = await browser.autonomous_research(
            topic="AI security",
            depth="quick"
        )
        print(f"✅ Research completed: {result['phases_completed']} phases")
        
        return True
    except Exception as e:
        print(f"❌ Agentic AI failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🚀 TESTING ALL NEW FEATURES")
    print("=" * 80)
    
    results = {}
    
    results['analytics'] = test_analytics()
    results['rotation'] = test_model_rotation()
    results['forensics'] = test_forensic_models()
    results['agentic'] = await test_agentic_ai()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    for feature, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {feature.upper()}")
    
    total = sum(results.values())
    print(f"\n🎯 Total: {total}/{len(results)} tests passed")
    
    if total == len(results):
        print("\n🎉 ALL FEATURES WORKING! Your AGI system is ready!")
        print("\nNext steps:")
        print("1. Start continuous learning: python scripts/auto_continuous_learning.py")
        print("2. Or use Colab: Upload COLAB_CONTINUOUS_LEARNING.ipynb")
        print("3. Check analytics: python -c 'from council.data_analytics import get_analytics; get_analytics().print_daily_summary()'")
    else:
        print("\n⚠️ Some features need attention. Check errors above.")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
