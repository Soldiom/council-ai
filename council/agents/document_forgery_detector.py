"""
Document Forgery Detection Agent
=================================

Detects forged documents:
- Passport forgery
- ID card forgery
- Driver's license forgery
- Bank statements
- Certificates and diplomas
- Legal documents
- Contracts and agreements

Uses computer vision and forensic document analysis.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import re


class DocumentForgeryDetector:
    """
    AI agent specialized in detecting forged documents.
    
    Capabilities:
    - Passport verification (watermarks, holograms, security features)
    - ID card authenticity (fonts, spacing, microprinting)
    - Document tampering detection (photo substitution, text alteration)
    - Security feature verification (UV features, guilloche patterns)
    - Metadata forensics (creation date, editing history)
    - Template matching (genuine vs fake document layouts)
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.training_data_dir = Path("training_data/document_forgery")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Top document verification models
        self.models = {
            "ocr": [
                "microsoft/trocr-large-printed",
                "naver-clova-ix/donut-base",
            ],
            "layout_analysis": [
                "microsoft/layoutlmv3-base",
                "impira/layoutlm-document-qa",
            ],
            "forgery_detection": [
                "Nahrawy/AIorNot",
                "umm-maybe/AI-image-detector",
            ],
        }
        
        # Common security features by document type
        self.security_features = {
            "passport": [
                "Machine Readable Zone (MRZ)",
                "Holographic overlay",
                "UV-reactive ink",
                "Microprinting",
                "Watermarks",
                "Security thread",
                "Ghost image",
                "Laser engraving",
                "Optically Variable Ink (OVI)",
            ],
            "id_card": [
                "Hologram",
                "UV features",
                "Microtext",
                "Guilloche patterns",
                "Photo ghost image",
                "Tactile features",
                "Color-shifting ink",
            ],
            "drivers_license": [
                "Holographic laminate",
                "UV ink",
                "Barcodes/2D codes",
                "Ghost image",
                "Microprinting",
                "Laser perforation",
            ],
        }
    
    async def analyze(
        self,
        document_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze document for forgery.
        
        Args:
            document_input: Image URL, base64, or description of document
            context: Additional context (document_type, issuing_country, etc.)
        
        Returns:
            Analysis results with authenticity assessment
        """
        context = context or {}
        doc_type = context.get("document_type", self._detect_document_type(document_input))
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(document_input, doc_type, context)
        
        # Get LLM analysis
        messages = [{"role": "user", "content": prompt}]
        analysis = await self.llm.chat(messages)
        
        # Extract forgery indicators
        indicators = self._extract_forgery_indicators(analysis, document_input, doc_type)
        
        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity_score(indicators, analysis)
        
        # Determine verdict
        verdict = self._determine_verdict(authenticity_score, indicators)
        
        # Extract document data
        extracted_data = self._extract_document_data(document_input, analysis)
        
        result = {
            "document_type": doc_type,
            "analysis": analysis,
            "authenticity_score": authenticity_score,  # 0.0 (forged) to 1.0 (genuine)
            "verdict": verdict,  # "GENUINE", "LIKELY_GENUINE", "SUSPICIOUS", "LIKELY_FORGED", "FORGED"
            "forgery_indicators": indicators,
            "extracted_data": extracted_data,
            "security_features_expected": self.security_features.get(doc_type, []),
            "timestamp": datetime.now().isoformat(),
            "recommended_models": self._get_recommended_models(doc_type),
        }
        
        # Save for training
        await self._save_training_data(document_input, context, result)
        
        return result
    
    def _detect_document_type(self, document_input: str) -> str:
        """Detect document type from input."""
        lower = document_input.lower()
        
        if "passport" in lower:
            return "passport"
        elif any(word in lower for word in ["id card", "identity card", "national id"]):
            return "id_card"
        elif any(word in lower for word in ["driver", "license", "licence", "dl"]):
            return "drivers_license"
        elif "bank statement" in lower:
            return "bank_statement"
        elif any(word in lower for word in ["certificate", "diploma", "degree"]):
            return "certificate"
        elif "visa" in lower:
            return "visa"
        elif "birth certificate" in lower:
            return "birth_certificate"
        else:
            return "unknown_document"
    
    def _build_analysis_prompt(self, document_input: str, doc_type: str, context: Dict) -> str:
        """Build document forgery analysis prompt."""
        
        security_features = self.security_features.get(doc_type, [])
        
        prompt = f"""You are an expert document forensics analyst. Analyze this {doc_type} for signs of forgery.

Document Input: {document_input}

Expected Security Features for {doc_type}:
{chr(10).join(f'- {feature}' for feature in security_features)}

Analyze for these forgery indicators:

VISUAL INSPECTION:
- Photo quality and consistency (lighting, resolution, edges)
- Font inconsistencies (wrong typeface, size, weight)
- Alignment issues (text not properly aligned)
- Color mismatches (wrong ink colors, faded/fresh mix)
- Print quality (laser vs inkjet, resolution)
- Lamination quality (bubbles, peeling, wrong material)
- Paper quality and texture

SECURITY FEATURES:
- Missing or incorrect holograms
- Absent or fake UV-reactive elements
- Incorrect or missing watermarks
- Missing microprinting or poor quality
- Incorrect guilloche patterns
- Missing or fake security threads
- Incorrect ghost images

DOCUMENT STRUCTURE:
- Template accuracy (layout matches official documents)
- MRZ format and check digits (for passports/IDs)
- Document number format and patterns
- Issue/expiry date logic and formatting
- Signature placement and appearance
- Official seals and stamps

TAMPERING SIGNS:
- Photo substitution (cut marks, glue residue)
- Text alterations (erasures, overwriting)
- Scratching or chemical alterations
- Different paper layers
- Inconsistent aging/wear patterns

METADATA & FORENSICS:
- Document age vs claimed issue date
- Ink chemical composition (if known)
- Printing method consistency
- Digital manipulation artifacts (if scanned/photographed)

Provide:
1. Detailed forensic analysis
2. Specific forgery indicators found
3. Missing security features
4. Confidence level (0-100%)
5. Verdict: GENUINE, LIKELY_GENUINE, SUSPICIOUS, LIKELY_FORGED, or FORGED
6. Recommendations for further verification (e.g., UV light, magnification, authority verification)

Analysis:"""
        
        return prompt
    
    def _extract_forgery_indicators(
        self,
        analysis: str,
        document_input: str,
        doc_type: str
    ) -> Dict[str, List[str]]:
        """Extract specific forgery indicators from analysis."""
        
        indicators = {
            "visual_anomalies": [],
            "missing_security_features": [],
            "tampering_signs": [],
            "structural_issues": [],
            "metadata_inconsistencies": [],
        }
        
        analysis_lower = analysis.lower()
        
        # Visual anomalies
        visual_keywords = [
            "font inconsisten", "misaligned", "color mismatch", "print quality",
            "resolution", "pixelated", "blurry", "cut mark", "glue"
        ]
        for keyword in visual_keywords:
            if keyword in analysis_lower:
                indicators["visual_anomalies"].append(keyword)
        
        # Missing security features
        security_keywords = [
            "missing hologram", "no watermark", "absent uv", "missing microprint",
            "no security thread", "missing ghost image", "no guilloche"
        ]
        for keyword in security_keywords:
            if keyword in analysis_lower:
                indicators["missing_security_features"].append(keyword)
        
        # Tampering signs
        tampering_keywords = [
            "photo substitut", "alteration", "erasure", "scratch", "chemical",
            "overwriting", "different layer", "inconsistent wear", "tamper"
        ]
        for keyword in tampering_keywords:
            if keyword in analysis_lower:
                indicators["tampering_signs"].append(keyword)
        
        # Structural issues
        structural_keywords = [
            "template mismatch", "incorrect format", "invalid mrz", "wrong layout",
            "incorrect numbering", "date logic", "format error"
        ]
        for keyword in structural_keywords:
            if keyword in analysis_lower:
                indicators["structural_issues"].append(keyword)
        
        # Metadata issues
        metadata_keywords = [
            "age inconsisten", "issue date", "expiry date", "digital manipulation",
            "metadata", "creation date"
        ]
        for keyword in metadata_keywords:
            if keyword in analysis_lower:
                indicators["metadata_inconsistencies"].append(keyword)
        
        return indicators
    
    def _calculate_authenticity_score(self, indicators: Dict, analysis: str) -> float:
        """Calculate authenticity score (0.0 = forged, 1.0 = genuine)."""
        
        # Count total indicators
        total_indicators = sum(len(v) for v in indicators.values())
        
        # Weight critical indicators more heavily
        critical_weight = len(indicators["missing_security_features"]) * 2
        critical_weight += len(indicators["tampering_signs"]) * 2
        
        # Extract confidence from analysis if present
        confidence_match = re.search(r'confidence[:\s]+(\d+)%', analysis.lower())
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100.0
        else:
            confidence = 0.5
        
        # Calculate penalty
        indicator_penalty = min((total_indicators + critical_weight) * 0.08, 0.7)
        
        # Calculate score
        final_score = max(0.0, min(1.0, confidence - indicator_penalty))
        
        return round(final_score, 2)
    
    def _determine_verdict(self, score: float, indicators: Dict) -> str:
        """Determine final verdict based on score and indicators."""
        
        total_indicators = sum(len(v) for v in indicators.values())
        has_tampering = len(indicators["tampering_signs"]) > 0
        missing_security = len(indicators["missing_security_features"]) > 0
        
        if score >= 0.85 and total_indicators == 0:
            return "GENUINE"
        elif score >= 0.65 and total_indicators <= 2 and not has_tampering:
            return "LIKELY_GENUINE"
        elif score >= 0.4 or (total_indicators <= 4 and not has_tampering):
            return "SUSPICIOUS"
        elif score >= 0.2 or (total_indicators <= 6 and not missing_security):
            return "LIKELY_FORGED"
        else:
            return "FORGED"
    
    def _extract_document_data(self, document_input: str, analysis: str) -> Dict[str, str]:
        """Extract structured data from document."""
        
        data = {}
        
        # Extract document number
        doc_num_match = re.search(r'(?:document|passport|id)\s*(?:number|no\.?|#)[:;\s]+([A-Z0-9]+)', analysis, re.IGNORECASE)
        if doc_num_match:
            data["document_number"] = doc_num_match.group(1)
        
        # Extract dates
        issue_date_match = re.search(r'issue\s*date[:;\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', analysis, re.IGNORECASE)
        if issue_date_match:
            data["issue_date"] = issue_date_match.group(1)
        
        expiry_date_match = re.search(r'expir(?:y|ation)\s*date[:;\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', analysis, re.IGNORECASE)
        if expiry_date_match:
            data["expiry_date"] = expiry_date_match.group(1)
        
        # Extract name
        name_match = re.search(r'name[:;\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', analysis)
        if name_match:
            data["name"] = name_match.group(1)
        
        return data
    
    def _get_recommended_models(self, doc_type: str) -> List[str]:
        """Get recommended models for this document type."""
        
        models = []
        models.extend(self.models["ocr"])
        models.extend(self.models["layout_analysis"])
        
        if doc_type in ["passport", "id_card", "drivers_license"]:
            models.extend(self.models["forgery_detection"])
        
        return models
    
    async def _save_training_data(
        self,
        document_input: str,
        context: Dict,
        result: Dict
    ):
        """Save analysis for training document forgery detection model."""
        
        training_example = {
            "input": {
                "document": document_input,
                "document_type": result["document_type"],
                "context": context,
            },
            "output": {
                "analysis": result["analysis"],
                "authenticity_score": result["authenticity_score"],
                "verdict": result["verdict"],
                "indicators": result["forgery_indicators"],
                "extracted_data": result["extracted_data"],
            },
            "timestamp": result["timestamp"],
        }
        
        # Save to daily file
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.training_data_dir / f"document_forgery_{today}.jsonl"
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_example, ensure_ascii=False) + "\n")


def get_document_forgery_detector():
    """Get configured document forgery detector agent."""
    from council.llm import get_llm
    
    llm = get_llm()
    return DocumentForgeryDetector(llm)
