"""
🔬 FORENSIC AI MODELS CATALOG

Specialized models for forensic analysis:
- Audio: Whisper (transcription), VoxCeleb (speaker recognition)
- Images: Forensic image analysis and comparison
- Deepfake detection
- Document forgery detection
- Metadata analysis
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ForensicCapability(Enum):
    """Forensic analysis capabilities"""
    AUDIO_TRANSCRIPTION = "audio_transcription"
    SPEAKER_RECOGNITION = "speaker_recognition"
    VOICE_COMPARISON = "voice_comparison"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    AUDIO_AUTHENTICATION = "audio_authentication"
    
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_COMPARISON = "image_comparison"
    METADATA_EXTRACTION = "metadata_extraction"
    TAMPERING_DETECTION = "tampering_detection"
    DEEPFAKE_DETECTION = "deepfake_detection"
    
    DOCUMENT_ANALYSIS = "document_analysis"
    SIGNATURE_VERIFICATION = "signature_verification"
    FONT_ANALYSIS = "font_analysis"
    
    VIDEO_ANALYSIS = "video_analysis"
    FACE_RECOGNITION = "face_recognition"


@dataclass
class ForensicModel:
    """Configuration for forensic AI model"""
    name: str
    provider: str
    model_id: str
    capabilities: List[ForensicCapability]
    
    # Performance metrics
    accuracy: float  # 0-1
    speed: str  # "fast", "medium", "slow"
    
    # Availability
    open_source: bool
    huggingface_available: bool
    
    # Use case
    best_for: str
    cost: str  # "free", "low", "medium", "high"


# ========================================
# AUDIO FORENSIC MODELS
# ========================================

AUDIO_MODELS = {
    "whisper-large-v3": ForensicModel(
        name="Whisper Large v3",
        provider="OpenAI",
        model_id="openai/whisper-large-v3",
        capabilities=[
            ForensicCapability.AUDIO_TRANSCRIPTION,
            ForensicCapability.AUDIO_ENHANCEMENT
        ],
        accuracy=0.96,
        speed="medium",
        open_source=True,
        huggingface_available=True,
        best_for="High-accuracy audio transcription, multilingual support (99 languages)",
        cost="free"
    ),
    
    "whisper-medium": ForensicModel(
        name="Whisper Medium",
        provider="OpenAI",
        model_id="openai/whisper-medium",
        capabilities=[ForensicCapability.AUDIO_TRANSCRIPTION],
        accuracy=0.92,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Faster transcription with good accuracy",
        cost="free"
    ),
    
    "voxceleb-resnet": ForensicModel(
        name="VoxCeleb ResNet",
        provider="VoxCeleb",
        model_id="microsoft/wavlm-base-plus-sv",
        capabilities=[
            ForensicCapability.SPEAKER_RECOGNITION,
            ForensicCapability.VOICE_COMPARISON
        ],
        accuracy=0.94,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Speaker identification and verification",
        cost="free"
    ),
    
    "speechbrain-speaker": ForensicModel(
        name="SpeechBrain Speaker Recognition",
        provider="SpeechBrain",
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        capabilities=[
            ForensicCapability.SPEAKER_RECOGNITION,
            ForensicCapability.VOICE_COMPARISON
        ],
        accuracy=0.93,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Embedding-based speaker comparison",
        cost="free"
    ),
    
    "wav2vec2-speaker": ForensicModel(
        name="Wav2Vec2 Speaker Verification",
        provider="Facebook",
        model_id="facebook/wav2vec2-base-960h",
        capabilities=[
            ForensicCapability.SPEAKER_RECOGNITION,
            ForensicCapability.AUDIO_AUTHENTICATION
        ],
        accuracy=0.91,
        speed="medium",
        open_source=True,
        huggingface_available=True,
        best_for="Audio feature extraction for forensics",
        cost="free"
    ),
    
    "audio-deepfake-detector": ForensicModel(
        name="Audio Deepfake Detector",
        provider="Various",
        model_id="openai/whisper-large-v3",  # Can detect synthetic speech
        capabilities=[
            ForensicCapability.AUDIO_AUTHENTICATION,
            ForensicCapability.DEEPFAKE_DETECTION
        ],
        accuracy=0.88,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Detecting AI-generated or manipulated audio",
        cost="free"
    ),
}


# ========================================
# IMAGE FORENSIC MODELS
# ========================================

IMAGE_MODELS = {
    "clip-forensics": ForensicModel(
        name="CLIP for Forensic Analysis",
        provider="OpenAI",
        model_id="openai/clip-vit-large-patch14",
        capabilities=[
            ForensicCapability.IMAGE_ANALYSIS,
            ForensicCapability.IMAGE_COMPARISON,
            ForensicCapability.DEEPFAKE_DETECTION
        ],
        accuracy=0.89,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Image-text matching, visual similarity for evidence",
        cost="free"
    ),
    
    "exif-analyzer": ForensicModel(
        name="EXIF Metadata Analyzer",
        provider="Custom",
        model_id="metadata/exiftool",
        capabilities=[
            ForensicCapability.METADATA_EXTRACTION,
            ForensicCapability.TAMPERING_DETECTION
        ],
        accuracy=0.99,
        speed="fast",
        open_source=True,
        huggingface_available=False,
        best_for="Extracting and analyzing image metadata for authenticity",
        cost="free"
    ),
    
    "error-level-analysis": ForensicModel(
        name="Error Level Analysis (ELA)",
        provider="Custom",
        model_id="forensic/ela",
        capabilities=[
            ForensicCapability.TAMPERING_DETECTION,
            ForensicCapability.IMAGE_ANALYSIS
        ],
        accuracy=0.85,
        speed="fast",
        open_source=True,
        huggingface_available=False,
        best_for="Detecting image manipulation and editing",
        cost="free"
    ),
    
    "face-recognition": ForensicModel(
        name="DeepFace Recognition",
        provider="Various",
        model_id="buffalo_l",
        capabilities=[
            ForensicCapability.FACE_RECOGNITION,
            ForensicCapability.IMAGE_COMPARISON
        ],
        accuracy=0.97,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Face matching and identification in evidence",
        cost="free"
    ),
    
    "image-deepfake-detector": ForensicModel(
        name="Image Deepfake Detector",
        provider="Various",
        model_id="umm-maybe/AI-image-detector",
        capabilities=[
            ForensicCapability.DEEPFAKE_DETECTION,
            ForensicCapability.IMAGE_ANALYSIS
        ],
        accuracy=0.92,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Detecting AI-generated images (DALL-E, Midjourney, etc.)",
        cost="free"
    ),
}


# ========================================
# VIDEO FORENSIC MODELS
# ========================================

VIDEO_MODELS = {
    "video-deepfake-detector": ForensicModel(
        name="Video Deepfake Detector",
        provider="Various",
        model_id="selimsef/dfdc_deepfake_challenge",
        capabilities=[
            ForensicCapability.DEEPFAKE_DETECTION,
            ForensicCapability.VIDEO_ANALYSIS,
            ForensicCapability.FACE_RECOGNITION
        ],
        accuracy=0.87,
        speed="slow",
        open_source=True,
        huggingface_available=True,
        best_for="Detecting deepfake videos",
        cost="free"
    ),
    
    "face-swap-detector": ForensicModel(
        name="Face Swap Detector",
        provider="Various",
        model_id="forensic/face-swap",
        capabilities=[
            ForensicCapability.DEEPFAKE_DETECTION,
            ForensicCapability.FACE_RECOGNITION
        ],
        accuracy=0.90,
        speed="medium",
        open_source=True,
        huggingface_available=True,
        best_for="Detecting face replacement in videos",
        cost="free"
    ),
}


# ========================================
# DOCUMENT FORENSIC MODELS
# ========================================

DOCUMENT_MODELS = {
    "document-ai": ForensicModel(
        name="Document AI OCR",
        provider="Google",
        model_id="google/documentai",
        capabilities=[
            ForensicCapability.DOCUMENT_ANALYSIS,
            ForensicCapability.METADATA_EXTRACTION
        ],
        accuracy=0.95,
        speed="fast",
        open_source=False,
        huggingface_available=False,
        best_for="OCR and document structure analysis",
        cost="medium"
    ),
    
    "signature-verification": ForensicModel(
        name="Signature Verification CNN",
        provider="Various",
        model_id="forensic/signature-verify",
        capabilities=[
            ForensicCapability.SIGNATURE_VERIFICATION,
            ForensicCapability.DOCUMENT_ANALYSIS
        ],
        accuracy=0.91,
        speed="fast",
        open_source=True,
        huggingface_available=True,
        best_for="Verifying signature authenticity",
        cost="free"
    ),
    
    "font-forensics": ForensicModel(
        name="Font Analysis Model",
        provider="Custom",
        model_id="forensic/font-analysis",
        capabilities=[
            ForensicCapability.FONT_ANALYSIS,
            ForensicCapability.DOCUMENT_ANALYSIS,
            ForensicCapability.TAMPERING_DETECTION
        ],
        accuracy=0.88,
        speed="medium",
        open_source=True,
        huggingface_available=False,
        best_for="Detecting font inconsistencies in forged documents",
        cost="free"
    ),
}


# ========================================
# ALL FORENSIC MODELS COMBINED
# ========================================

ALL_FORENSIC_MODELS = {
    **AUDIO_MODELS,
    **IMAGE_MODELS,
    **VIDEO_MODELS,
    **DOCUMENT_MODELS,
}


# ========================================
# HELPER FUNCTIONS
# ========================================

def get_models_by_capability(capability: ForensicCapability) -> List[ForensicModel]:
    """Get all models that support a specific capability"""
    return [
        model for model in ALL_FORENSIC_MODELS.values()
        if capability in model.capabilities
    ]


def get_best_model_for_task(task_type: str) -> ForensicModel:
    """Get the best model for a specific forensic task"""
    task_mapping = {
        # Audio tasks
        "transcribe_audio": AUDIO_MODELS["whisper-large-v3"],
        "identify_speaker": AUDIO_MODELS["voxceleb-resnet"],
        "compare_voices": AUDIO_MODELS["speechbrain-speaker"],
        "detect_audio_deepfake": AUDIO_MODELS["audio-deepfake-detector"],
        
        # Image tasks
        "analyze_image": IMAGE_MODELS["clip-forensics"],
        "extract_metadata": IMAGE_MODELS["exif-analyzer"],
        "detect_tampering": IMAGE_MODELS["error-level-analysis"],
        "recognize_face": IMAGE_MODELS["face-recognition"],
        "detect_image_deepfake": IMAGE_MODELS["image-deepfake-detector"],
        
        # Video tasks
        "detect_video_deepfake": VIDEO_MODELS["video-deepfake-detector"],
        "detect_face_swap": VIDEO_MODELS["face-swap-detector"],
        
        # Document tasks
        "verify_signature": DOCUMENT_MODELS["signature-verification"],
        "analyze_font": DOCUMENT_MODELS["font-forensics"],
    }
    
    return task_mapping.get(task_type, AUDIO_MODELS["whisper-large-v3"])


def print_forensic_catalog():
    """Print complete forensic model catalog"""
    print("\n" + "=" * 80)
    print("🔬 FORENSIC AI MODELS CATALOG")
    print("=" * 80)
    
    print("\n🎤 AUDIO MODELS (Whisper, VoxCeleb, Speaker Recognition)")
    print("-" * 80)
    for name, model in AUDIO_MODELS.items():
        print(f"\n{model.name}")
        print(f"   Model ID: {model.model_id}")
        print(f"   Accuracy: {model.accuracy * 100:.1f}%")
        print(f"   Speed: {model.speed}")
        print(f"   Cost: {model.cost}")
        print(f"   Best for: {model.best_for}")
        print(f"   Open Source: {'✅' if model.open_source else '❌'}")
        print(f"   HuggingFace: {'✅' if model.huggingface_available else '❌'}")
    
    print("\n\n🖼️  IMAGE MODELS (Forensic Analysis, Deepfake Detection)")
    print("-" * 80)
    for name, model in IMAGE_MODELS.items():
        print(f"\n{model.name}")
        print(f"   Model ID: {model.model_id}")
        print(f"   Accuracy: {model.accuracy * 100:.1f}%")
        print(f"   Best for: {model.best_for}")
    
    print("\n\n🎬 VIDEO MODELS (Deepfake Detection, Face Swap)")
    print("-" * 80)
    for name, model in VIDEO_MODELS.items():
        print(f"\n{model.name}")
        print(f"   Model ID: {model.model_id}")
        print(f"   Accuracy: {model.accuracy * 100:.1f}%")
        print(f"   Best for: {model.best_for}")
    
    print("\n\n📄 DOCUMENT MODELS (Signature, Font Analysis)")
    print("-" * 80)
    for name, model in DOCUMENT_MODELS.items():
        print(f"\n{model.name}")
        print(f"   Model ID: {model.model_id}")
        print(f"   Accuracy: {model.accuracy * 100:.1f}%")
        print(f"   Best for: {model.best_for}")
    
    print("\n" + "=" * 80)
    print(f"Total Forensic Models: {len(ALL_FORENSIC_MODELS)}")
    print("=" * 80 + "\n")


# ========================================
# FORENSIC DATASETS
# ========================================

FORENSIC_DATASETS = {
    # Audio datasets
    "voxceleb1": {
        "name": "VoxCeleb1",
        "type": "audio",
        "size": "100K+ videos, 1,251 celebrities",
        "use": "Speaker recognition training",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html"
    },
    "voxceleb2": {
        "name": "VoxCeleb2",
        "type": "audio",
        "size": "1M+ videos, 6,112 identities",
        "use": "Large-scale speaker recognition",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html"
    },
    "asvspoof": {
        "name": "ASVspoof",
        "type": "audio",
        "size": "100K+ samples",
        "use": "Audio deepfake detection",
        "url": "https://www.asvspoof.org/"
    },
    
    # Image datasets
    "casia": {
        "name": "CASIA Image Tampering Dataset",
        "type": "image",
        "size": "12,614 images",
        "use": "Image forgery detection training",
        "url": "https://github.com/namtpham/casia2groundtruth"
    },
    "nist": {
        "name": "NIST Image Forensics",
        "type": "image",
        "size": "5,000+ images",
        "use": "Professional forensic image analysis",
        "url": "https://www.nist.gov/itl/iad/mig/nimble-challenge"
    },
    "ffhq": {
        "name": "FFHQ (Flickr-Faces-HQ)",
        "type": "image",
        "size": "70K high-quality faces",
        "use": "Face recognition and deepfake training",
        "url": "https://github.com/NVlabs/ffhq-dataset"
    },
    
    # Video datasets
    "dfdc": {
        "name": "DeepFake Detection Challenge",
        "type": "video",
        "size": "100K+ videos",
        "use": "Deepfake detection training",
        "url": "https://www.kaggle.com/c/deepfake-detection-challenge"
    },
    "faceforensics": {
        "name": "FaceForensics++",
        "type": "video",
        "size": "1,000+ original videos, 5,000+ manipulated",
        "use": "Face manipulation detection",
        "url": "https://github.com/ondyari/FaceForensics"
    },
}


if __name__ == "__main__":
    # Test forensic catalog
    print_forensic_catalog()
    
    print("\n📊 BEST MODELS FOR COMMON TASKS:")
    print("-" * 80)
    tasks = [
        "transcribe_audio",
        "identify_speaker",
        "detect_audio_deepfake",
        "detect_image_deepfake",
        "verify_signature"
    ]
    
    for task in tasks:
        model = get_best_model_for_task(task)
        print(f"\n{task.replace('_', ' ').title()}:")
        print(f"   → {model.name} ({model.accuracy * 100:.1f}% accuracy)")
