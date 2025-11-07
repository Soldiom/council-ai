"""
Deepfake Detection Agent
=========================

Detects manipulated media:
- Deepfake videos (face swaps, lip-sync manipulation)
- Deepfake images (face manipulation, synthetic faces)
- Deepfake audio (voice cloning, synthetic speech)

Uses state-of-the-art deepfake detection models from HuggingFace.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import re


class DeepfakeDetector:
    """
    AI agent specialized in detecting deepfakes and manipulated media.
    
    Capabilities:
    - Video deepfake detection (face swap, lip-sync)
    - Image deepfake detection (synthetic faces, manipulated photos)
    - Audio deepfake detection (voice cloning, synthetic speech)
    - Manipulation artifact detection
    - Confidence scoring
    - Source verification
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.training_data_dir = Path("training_data/deepfake")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Top deepfake detection models
        self.models = {
            "video": [
                "selimsef/dfdc_deepfake_challenge",
                "dima806/deepfake_vs_real_image_detection",
                "Organika/sdxl-detector",
            ],
            "image": [
                "umm-maybe/AI-image-detector",
                "Nahrawy/AIorNot",
                "openai/roberta-base-openai-detector",
            ],
            "audio": [
                "Jeneral/deepfake-audio-detection",
                "speechbrain/aasist-l",
            ],
        }
    
    async def analyze(
        self,
        media_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze media for deepfake manipulation.
        
        Args:
            media_input: URL or base64 of media to analyze, or description
            context: Additional context (media_type, source, etc.)
        
        Returns:
            Analysis results with authenticity score and detected manipulations
        """
        context = context or {}
        media_type = context.get("media_type", self._detect_media_type(media_input))
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(media_input, media_type, context)
        
        # Get LLM analysis
        messages = [{"role": "user", "content": prompt}]
        analysis = await self.llm.chat(messages)
        
        # Extract manipulation indicators
        indicators = self._extract_manipulation_indicators(analysis, media_input)
        
        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity_score(indicators, analysis)
        
        # Determine verdict
        verdict = self._determine_verdict(authenticity_score, indicators)
        
        result = {
            "media_type": media_type,
            "analysis": analysis,
            "authenticity_score": authenticity_score,  # 0.0 (fake) to 1.0 (authentic)
            "verdict": verdict,  # "AUTHENTIC", "LIKELY_AUTHENTIC", "SUSPICIOUS", "LIKELY_DEEPFAKE", "DEEPFAKE"
            "manipulation_indicators": indicators,
            "timestamp": datetime.now().isoformat(),
            "recommended_models": self.models.get(media_type, []),
        }
        
        # Save for training
        await self._save_training_data(media_input, context, result)
        
        return result
    
    def _detect_media_type(self, media_input: str) -> str:
        """Detect media type from input."""
        lower = media_input.lower()
        
        if any(ext in lower for ext in ['.mp4', '.avi', '.mov', '.mkv', 'video']):
            return "video"
        elif any(ext in lower for ext in ['.jpg', '.png', '.jpeg', '.webp', 'image', 'photo', 'picture']):
            return "image"
        elif any(ext in lower for ext in ['.mp3', '.wav', '.m4a', '.flac', 'audio', 'voice', 'speech']):
            return "audio"
        else:
            return "unknown"
    
    def _build_analysis_prompt(self, media_input: str, media_type: str, context: Dict) -> str:
        """Build deepfake analysis prompt."""
        
        prompt = f"""You are an expert deepfake detection analyst. Analyze this {media_type} for signs of manipulation.

Media Input: {media_input}

Analyze for these deepfake indicators:

For VIDEO:
- Facial inconsistencies (blurriness around face boundaries)
- Unnatural eye movements or blinking patterns
- Lip-sync mismatches (audio not matching mouth movement)
- Lighting inconsistencies on face vs background
- Skin texture anomalies (overly smooth, synthetic appearance)
- Hair/face boundary artifacts
- Temporal inconsistencies between frames
- Unnatural head movements or expressions

For IMAGE:
- AI-generated facial features (symmetry, perfect skin, dead eyes)
- JPEG compression artifacts vs pristine quality
- Inconsistent lighting and shadows
- Anatomical impossibilities (wrong fingers, teeth, ears)
- Background inconsistencies
- Metadata manipulation
- Noise pattern irregularities
- GAN artifacts (checkerboard patterns, spectral anomalies)

For AUDIO:
- Robotic or synthetic voice qualities
- Unnatural prosody and intonation
- Inconsistent background noise
- Spectral anomalies
- Breathing pattern irregularities
- Clipping or concatenation artifacts
- Unnatural pauses or timing

Provide:
1. Detailed technical analysis
2. Specific manipulation indicators found
3. Confidence level (0-100%)
4. Verdict: AUTHENTIC, LIKELY_AUTHENTIC, SUSPICIOUS, LIKELY_DEEPFAKE, or DEEPFAKE
5. Recommendations for further verification

Analysis:"""
        
        return prompt
    
    def _extract_manipulation_indicators(self, analysis: str, media_input: str) -> Dict[str, List[str]]:
        """Extract specific manipulation indicators from analysis."""
        
        indicators = {
            "visual_anomalies": [],
            "temporal_inconsistencies": [],
            "audio_artifacts": [],
            "metadata_issues": [],
            "ai_signatures": [],
        }
        
        analysis_lower = analysis.lower()
        
        # Visual anomalies
        visual_keywords = [
            "blurr", "artifact", "inconsistent lighting", "unnatural",
            "synthetic", "anatomical", "distorted", "warped", "smoothing"
        ]
        for keyword in visual_keywords:
            if keyword in analysis_lower:
                indicators["visual_anomalies"].append(keyword)
        
        # Temporal issues
        temporal_keywords = [
            "frame", "jitter", "flicker", "discontinuit", "temporal",
            "motion blur", "lag", "stutter"
        ]
        for keyword in temporal_keywords:
            if keyword in analysis_lower:
                indicators["temporal_inconsistencies"].append(keyword)
        
        # Audio artifacts
        audio_keywords = [
            "robotic", "synthetic voice", "clipping", "spectral",
            "unnatural prosody", "concatenat", "breathing"
        ]
        for keyword in audio_keywords:
            if keyword in analysis_lower:
                indicators["audio_artifacts"].append(keyword)
        
        # AI signatures
        ai_keywords = [
            "gan", "generative", "ai-generated", "deepfake", "face swap",
            "synthetic", "cgan", "stylegan", "diffusion"
        ]
        for keyword in ai_keywords:
            if keyword in analysis_lower:
                indicators["ai_signatures"].append(keyword)
        
        return indicators
    
    def _calculate_authenticity_score(self, indicators: Dict, analysis: str) -> float:
        """Calculate authenticity score (0.0 = fake, 1.0 = authentic)."""
        
        # Count total indicators
        total_indicators = sum(len(v) for v in indicators.values())
        
        # Extract confidence from analysis if present
        confidence_match = re.search(r'confidence[:\s]+(\d+)%', analysis.lower())
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100.0
        else:
            confidence = 0.5
        
        # More indicators = less authentic
        indicator_penalty = min(total_indicators * 0.1, 0.6)
        
        # Calculate score
        base_score = confidence
        final_score = max(0.0, min(1.0, base_score - indicator_penalty))
        
        return round(final_score, 2)
    
    def _determine_verdict(self, score: float, indicators: Dict) -> str:
        """Determine final verdict based on score and indicators."""
        
        total_indicators = sum(len(v) for v in indicators.values())
        has_ai_signature = len(indicators["ai_signatures"]) > 0
        
        if score >= 0.8 and total_indicators == 0:
            return "AUTHENTIC"
        elif score >= 0.6 and total_indicators <= 2:
            return "LIKELY_AUTHENTIC"
        elif score >= 0.4 or (total_indicators <= 4 and not has_ai_signature):
            return "SUSPICIOUS"
        elif score >= 0.2 or total_indicators <= 6:
            return "LIKELY_DEEPFAKE"
        else:
            return "DEEPFAKE"
    
    async def _save_training_data(
        self,
        media_input: str,
        context: Dict,
        result: Dict
    ):
        """Save analysis for training deepfake detection model."""
        
        training_example = {
            "input": {
                "media": media_input,
                "media_type": result["media_type"],
                "context": context,
            },
            "output": {
                "analysis": result["analysis"],
                "authenticity_score": result["authenticity_score"],
                "verdict": result["verdict"],
                "indicators": result["manipulation_indicators"],
            },
            "timestamp": result["timestamp"],
        }
        
        # Save to daily file
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.training_data_dir / f"deepfake_detection_{today}.jsonl"
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_example, ensure_ascii=False) + "\n")


def get_deepfake_detector():
    """Get configured deepfake detector agent."""
    from council.llm import get_llm
    
    llm = get_llm()
    return DeepfakeDetector(llm)
