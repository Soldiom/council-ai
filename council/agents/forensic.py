"""
Forensic AI Agent - Specialized for Digital Forensics
======================================================

Analyzes digital evidence, logs, network traffic, malware, etc.
Collects training data specifically for forensics use cases.

This agent will:
1. Analyze security logs, malware samples, network traffic
2. Detect anomalies and threats
3. Generate forensic reports
4. Collect ALL forensic data for training YOUR forensic model
"""

from council.agents.base import BaseAgent
from typing import Dict, Any, Optional
from datetime import datetime
import json
import re
from pathlib import Path


class ForensicAgent(BaseAgent):
    """
    Specialized forensic analysis agent.
    
    Capabilities:
    - Log analysis (system, security, application logs)
    - Malware detection and analysis
    - Network traffic analysis
    - Threat intelligence
    - Incident response
    - Evidence collection
    - Timeline reconstruction
    - IOC (Indicator of Compromise) extraction
    """
    
    def __init__(self, llm=None):
        super().__init__(name="forensic", llm=llm)
        self.forensic_data_dir = Path("training_data/forensic")
        self.forensic_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Forensic knowledge base
        self.ioc_patterns = {
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "md5": r'\b[a-fA-F0-9]{32}\b',
            "sha1": r'\b[a-fA-F0-9]{40}\b',
            "sha256": r'\b[a-fA-F0-9]{64}\b',
            "cve": r'CVE-\d{4}-\d{4,7}',
        }
    
    async def analyze(self, input_data: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze forensic evidence.
        
        Args:
            input_data: Evidence to analyze (logs, malware, network data, etc.)
            context: Additional context (case_id, evidence_type, etc.)
        """
        analysis_type = self._detect_analysis_type(input_data, context)
        
        # Extract IOCs
        iocs = self._extract_iocs(input_data)
        
        # Perform specialized analysis
        if analysis_type == "log_analysis":
            result = await self._analyze_logs(input_data, iocs)
        elif analysis_type == "malware_analysis":
            result = await self._analyze_malware(input_data, iocs)
        elif analysis_type == "network_analysis":
            result = await self._analyze_network(input_data, iocs)
        elif analysis_type == "threat_intelligence":
            result = await self._threat_intel(input_data, iocs)
        else:
            result = await self._general_forensic_analysis(input_data, iocs)
        
        # Add metadata
        result["analysis_type"] = analysis_type
        result["iocs_found"] = iocs
        result["timestamp"] = datetime.now().isoformat()
        
        # Save as training data
        await self._save_forensic_training_data(input_data, result, context)
        
        return result
    
    def _detect_analysis_type(self, data: str, context: Optional[Dict] = None) -> str:
        """Detect type of forensic analysis needed."""
        data_lower = data.lower()
        
        if context and "analysis_type" in context:
            return context["analysis_type"]
        
        # Pattern matching
        if any(word in data_lower for word in ["error", "warning", "critical", "alert", "failed login"]):
            return "log_analysis"
        elif any(word in data_lower for word in ["malware", "virus", "trojan", "ransomware", "exploit"]):
            return "malware_analysis"
        elif any(word in data_lower for word in ["packet", "traffic", "connection", "port", "protocol"]):
            return "network_analysis"
        elif any(word in data_lower for word in ["threat", "vulnerability", "cve", "attack", "breach"]):
            return "threat_intelligence"
        else:
            return "general_forensic"
    
    def _extract_iocs(self, data: str) -> Dict[str, list]:
        """Extract Indicators of Compromise."""
        iocs = {}
        
        for ioc_type, pattern in self.ioc_patterns.items():
            matches = re.findall(pattern, data)
            if matches:
                iocs[ioc_type] = list(set(matches))  # Deduplicate
        
        return iocs
    
    async def _analyze_logs(self, logs: str, iocs: Dict) -> Dict[str, Any]:
        """Analyze security logs."""
        prompt = f"""You are a forensic log analyst. Analyze these logs for security incidents.

Logs:
{logs[:2000]}  # Truncate for prompt

IOCs found: {json.dumps(iocs, indent=2)}

Provide:
1. Summary of suspicious activities
2. Severity assessment (Critical/High/Medium/Low)
3. Recommended actions
4. Timeline of events
5. Potential attack vectors

Analysis:"""
        
        response = await self.llm.generate(prompt)
        
        return {
            "analysis": response,
            "log_summary": self._summarize_logs(logs),
            "severity": self._assess_severity(logs, iocs),
        }
    
    async def _analyze_malware(self, data: str, iocs: Dict) -> Dict[str, Any]:
        """Analyze malware sample or report."""
        prompt = f"""You are a malware analyst. Analyze this malware evidence.

Evidence:
{data[:2000]}

IOCs found: {json.dumps(iocs, indent=2)}

Provide:
1. Malware family/type identification
2. Capabilities and behavior
3. Network indicators (C2 servers, etc.)
4. File indicators (hashes, names)
5. Mitigation recommendations

Analysis:"""
        
        response = await self.llm.generate(prompt)
        
        return {
            "analysis": response,
            "malware_type": self._classify_malware(data),
            "threat_level": "High",  # Malware is always high priority
        }
    
    async def _analyze_network(self, data: str, iocs: Dict) -> Dict[str, Any]:
        """Analyze network traffic."""
        prompt = f"""You are a network forensics analyst. Analyze this network traffic.

Traffic data:
{data[:2000]}

IOCs found: {json.dumps(iocs, indent=2)}

Provide:
1. Suspicious connections
2. Data exfiltration indicators
3. Command & Control (C2) communication
4. Port scanning or reconnaissance
5. Recommended firewall rules

Analysis:"""
        
        response = await self.llm.generate(prompt)
        
        return {
            "analysis": response,
            "suspicious_ips": iocs.get("ip_address", []),
            "suspicious_urls": iocs.get("url", []),
        }
    
    async def _threat_intel(self, data: str, iocs: Dict) -> Dict[str, Any]:
        """Threat intelligence analysis."""
        prompt = f"""You are a threat intelligence analyst. Analyze this threat data.

Threat information:
{data[:2000]}

IOCs found: {json.dumps(iocs, indent=2)}

Provide:
1. Threat actor identification (if possible)
2. Attack methodology (TTPs)
3. Related CVEs or vulnerabilities
4. Attribution indicators
5. Defensive recommendations

Analysis:"""
        
        response = await self.llm.generate(prompt)
        
        return {
            "analysis": response,
            "cves_found": iocs.get("cve", []),
            "threat_level": self._assess_threat_level(data, iocs),
        }
    
    async def _general_forensic_analysis(self, data: str, iocs: Dict) -> Dict[str, Any]:
        """General forensic analysis."""
        prompt = f"""You are a digital forensics expert. Analyze this evidence.

Evidence:
{data[:2000]}

IOCs found: {json.dumps(iocs, indent=2)}

Provide comprehensive forensic analysis including:
1. Evidence summary
2. Key findings
3. Security implications
4. Recommendations
5. Next investigation steps

Analysis:"""
        
        response = await self.llm.generate(prompt)
        
        return {
            "analysis": response,
            "evidence_type": "general",
        }
    
    def _summarize_logs(self, logs: str) -> Dict[str, int]:
        """Quick log statistics."""
        return {
            "total_lines": len(logs.split('\n')),
            "errors": logs.lower().count('error'),
            "warnings": logs.lower().count('warning'),
            "failed_logins": logs.lower().count('failed login'),
        }
    
    def _assess_severity(self, data: str, iocs: Dict) -> str:
        """Assess severity based on IOCs and content."""
        score = 0
        
        # More IOCs = higher severity
        score += len(iocs.get("ip_address", [])) * 1
        score += len(iocs.get("url", [])) * 2
        score += len(iocs.get("md5", [])) * 3
        score += len(iocs.get("cve", [])) * 5
        
        # Keywords
        data_lower = data.lower()
        if "critical" in data_lower:
            score += 10
        if "exploit" in data_lower:
            score += 8
        if "breach" in data_lower:
            score += 15
        
        if score >= 20:
            return "Critical"
        elif score >= 10:
            return "High"
        elif score >= 5:
            return "Medium"
        else:
            return "Low"
    
    def _classify_malware(self, data: str) -> str:
        """Simple malware classification."""
        data_lower = data.lower()
        
        if "ransomware" in data_lower:
            return "Ransomware"
        elif "trojan" in data_lower:
            return "Trojan"
        elif "worm" in data_lower:
            return "Worm"
        elif "rootkit" in data_lower:
            return "Rootkit"
        else:
            return "Unknown/Generic"
    
    def _assess_threat_level(self, data: str, iocs: Dict) -> str:
        """Assess threat level."""
        if iocs.get("cve") or "exploit" in data.lower():
            return "Critical"
        elif "vulnerability" in data.lower():
            return "High"
        else:
            return "Medium"
    
    async def _save_forensic_training_data(
        self,
        input_data: str,
        result: Dict[str, Any],
        context: Optional[Dict] = None
    ):
        """Save forensic analysis as training data."""
        training_example = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": result,
            "context": context or {},
            "analysis_type": result.get("analysis_type"),
            "iocs": result.get("iocs_found", {}),
            "severity": result.get("severity", "Unknown"),
        }
        
        # Save to daily file
        date_str = datetime.now().strftime("%Y-%m-%d")
        forensic_file = self.forensic_data_dir / f"forensic_{date_str}.jsonl"
        
        with open(forensic_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_example) + "\n")
        
        print(f"💾 Saved forensic training data: {forensic_file}")


class ForensicModelTrainer:
    """Train specialized forensic AI model."""
    
    def __init__(self):
        self.data_dir = Path("training_data/forensic")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def prepare_forensic_dataset(self, output_file: str = "training_data/forensic_model.jsonl"):
        """Prepare forensic training dataset."""
        print("🔍 Preparing forensic training dataset...")
        
        # Collect all forensic data
        forensic_files = list(self.data_dir.glob("forensic_*.jsonl"))
        
        if not forensic_files:
            print("⚠️ No forensic data found yet")
            return None
        
        total_examples = 0
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as out:
            for file in forensic_files:
                with open(file, encoding="utf-8") as f:
                    for line in f:
                        example = json.loads(line)
                        
                        # Convert to training format
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an expert digital forensics analyst specializing in security incident analysis, malware detection, and threat intelligence."
                            },
                            {
                                "role": "user",
                                "content": f"Analyze this evidence:\n\n{example['input']}"
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps(example['output'], indent=2)
                            }
                        ]
                        
                        training_example = {"messages": messages}
                        out.write(json.dumps(training_example) + "\n")
                        total_examples += 1
        
        print(f"✅ Prepared {total_examples} forensic training examples")
        print(f"📁 Saved to: {output_path}")
        
        return {
            "total_examples": total_examples,
            "output_file": str(output_path),
            "ready_for_training": total_examples >= 50,  # Lower threshold for specialized model
        }
    
    async def train_forensic_model(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_name: str = "forensic-ai-model",
    ):
        """Train YOUR forensic AI model."""
        print("🎓 Training Forensic AI Model")
        print(f"   Base: {base_model}")
        print(f"   Output: aliAIML/{output_name}")
        print()
        
        # Prepare dataset
        dataset_info = await self.prepare_forensic_dataset()
        
        if not dataset_info:
            print("❌ No training data available")
            return
        
        if not dataset_info["ready_for_training"]:
            print(f"⚠️ Need at least 50 examples (have {dataset_info['total_examples']})")
            return
        
        print("🚀 Ready for fine-tuning!")
        print()
        print("📝 Training steps:")
        print("   1. Use Google Colab (FREE GPU)")
        print("   2. Upload:", dataset_info["output_file"])
        print("   3. Follow: COLAB_FINETUNING.md")
        print("   4. Result: aliAIML/forensic-ai-model")
        print()
        print("💡 This will create YOUR specialized forensic AI model!")
        
        return dataset_info


# Integration with main system
def get_forensic_agent():
    """Get or create forensic agent instance."""
    from council.llm import get_llm
    llm = get_llm(agent_name="forensic")
    return ForensicAgent(llm=llm)
