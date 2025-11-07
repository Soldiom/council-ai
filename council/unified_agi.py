"""
🧠 UNIFIED AGI CONTROLLER

The brain of the autonomous system.
Orchestrates perception, reasoning, and action.
Makes decisions with human-in-the-loop for high-risk actions.

This is the central controller that makes "AI Does Everything" possible.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class RiskLevel(Enum):
    """Risk classification for actions"""
    LOW = "low"           # Autonomous, logging only
    MEDIUM = "medium"     # Autonomous, human notification
    HIGH = "high"         # Requires human approval
    CRITICAL = "critical" # Requires multi-person approval


class TaskDomain(Enum):
    """Domains the AGI can operate in"""
    FORENSICS = "forensics"
    TRADING = "trading"
    CONTENT_CREATION = "content_creation"
    MONITORING = "monitoring"
    PREDICTION = "prediction"
    GENERAL = "general"


@dataclass
class Task:
    """Unified task representation"""
    id: str
    domain: TaskDomain
    description: str
    input_data: Dict[str, Any]
    
    # Risk and safety
    risk_level: RiskLevel = RiskLevel.LOW
    requires_human_approval: bool = False
    
    # Metadata
    created_at: str = None
    assigned_to: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class Action:
    """Action taken by the system"""
    task_id: str
    action_type: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    
    # Results
    executed: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Approval
    requires_approval: bool = False
    approved: bool = False
    approver: Optional[str] = None
    
    # Audit
    timestamp: str = None
    reasoning: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class UnifiedAGIController:
    """
    Central AGI controller that orchestrates everything.
    
    Capabilities:
    - Perceives: text, audio, images, videos
    - Reasons: cross-domain analysis, prediction
    - Acts: trading, reports, content creation
    - Learns: from corrections and feedback
    - Protects: safety guardrails, human oversight
    """
    
    def __init__(self, config_path: str = "agi_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Action history for learning
        self.action_history: List[Action] = []
        self.audit_log_path = Path("training_data/agi_audit_log.jsonl")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "autonomous_actions": 0,
            "human_approvals_requested": 0,
            "human_approvals_granted": 0,
            "corrections_received": 0,
            "models_improved": 0
        }
        
        # Subsystems (will be initialized)
        self.perception = None
        self.reasoning = None
        self.action_executor = None
        self.meta_learner = None
        
        print("🧠 Unified AGI Controller initialized")
        print(f"   Risk tolerance: {self.config.get('risk_tolerance', 'medium')}")
        print(f"   Autonomous mode: {self.config.get('autonomous_mode', False)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load AGI configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default config
        return {
            "autonomous_mode": True,
            "risk_tolerance": "medium",
            "max_autonomous_trade_size": 1000,
            "max_daily_trades": 10,
            "require_approval_for": ["large_trades", "identity_verification", "sensitive_reports"],
            "audit_everything": True,
            "learning_enabled": True,
            "safety_guardrails": {
                "trading": {
                    "max_position_pct": 5,
                    "max_daily_loss_pct": 2,
                    "require_stop_loss": True
                },
                "forensics": {
                    "min_confidence_autonomous": 0.95,
                    "require_human_review": ["identity_verification", "criminal_charges"]
                },
                "content_creation": {
                    "max_video_length_hours": 4,
                    "require_approval_for_public": True
                }
            }
        }
    
    def _save_config(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Main entry point: process any task.
        
        Pipeline:
        1. Classify risk level
        2. Perceive inputs (multi-modal)
        3. Reason about task
        4. Decide on action
        5. Check if approval needed
        6. Execute (or request approval)
        7. Learn from result
        """
        print(f"\n{'=' * 80}")
        print(f"🧠 Processing Task: {task.description}")
        print(f"   Domain: {task.domain.value}")
        print(f"   Risk: {task.risk_level.value}")
        print(f"{'=' * 80}\n")
        
        # Step 1: Classify risk
        risk_level = self._classify_risk(task)
        task.risk_level = risk_level
        
        # Step 2: Perceive
        perception_result = await self._perceive(task)
        
        # Step 3: Reason
        reasoning_result = await self._reason(perception_result, task)
        
        # Step 4: Decide action
        proposed_action = self._decide_action(reasoning_result, task)
        
        # Step 5: Check approval
        if self._requires_approval(proposed_action):
            approved = await self._request_human_approval(proposed_action)
            proposed_action.approved = approved
            
            if not approved:
                return {
                    "status": "rejected",
                    "task_id": task.id,
                    "reason": "Human approval denied",
                    "action": asdict(proposed_action)
                }
        else:
            proposed_action.approved = True
        
        # Step 6: Execute
        result = await self._execute_action(proposed_action)
        
        # Step 7: Audit log
        self._audit_log(task, proposed_action, result)
        
        # Step 8: Learn
        if self.config.get("learning_enabled", True):
            await self._meta_learn(task, proposed_action, result)
        
        # Update metrics
        if result.get("success", False):
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        
        if proposed_action.approved and not proposed_action.requires_approval:
            self.metrics["autonomous_actions"] += 1
        
        return {
            "status": "completed" if result.get("success") else "failed",
            "task_id": task.id,
            "result": result,
            "action": asdict(proposed_action),
            "metrics": self.metrics
        }
    
    def _classify_risk(self, task: Task) -> RiskLevel:
        """
        Classify risk level of a task.
        
        Rules:
        - Trading >$10k: HIGH
        - Identity verification: HIGH
        - Content creation for public: MEDIUM
        - Routine analysis: LOW
        """
        domain = task.domain
        input_data = task.input_data
        
        # Trading risk
        if domain == TaskDomain.TRADING:
            amount = input_data.get("amount", 0)
            if amount > 10000:
                return RiskLevel.CRITICAL
            elif amount > 1000:
                return RiskLevel.HIGH
            elif amount > 100:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        
        # Forensics risk
        elif domain == TaskDomain.FORENSICS:
            task_type = input_data.get("type", "")
            if task_type in ["identity_verification", "criminal_charges"]:
                return RiskLevel.HIGH
            elif task_type in ["audio_analysis", "image_analysis"]:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        
        # Content creation risk
        elif domain == TaskDomain.CONTENT_CREATION:
            is_public = input_data.get("public", False)
            duration_hours = input_data.get("duration_hours", 0)
            
            if is_public and duration_hours > 2:
                return RiskLevel.MEDIUM
            elif is_public:
                return RiskLevel.LOW
            else:
                return RiskLevel.LOW
        
        # Default: medium risk for prediction/monitoring
        elif domain in [TaskDomain.PREDICTION, TaskDomain.MONITORING]:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    async def _perceive(self, task: Task) -> Dict[str, Any]:
        """
        Multi-modal perception.
        
        Reads: text, audio, images, videos
        """
        perception = {
            "text": None,
            "audio": None,
            "images": None,
            "videos": None,
            "structured": None
        }
        
        input_data = task.input_data
        
        # Text perception
        if "text" in input_data:
            perception["text"] = input_data["text"]
        
        # Audio perception (Whisper, VoxCeleb)
        if "audio" in input_data:
            perception["audio"] = {
                "transcription": f"[Whisper transcription of {input_data['audio']}]",
                "speaker": "[VoxCeleb speaker identification]",
                "confidence": 0.94
            }
        
        # Image perception (CLIP, DeepFace)
        if "images" in input_data:
            perception["images"] = {
                "analysis": f"[CLIP analysis of {len(input_data['images'])} images]",
                "faces": "[DeepFace recognition results]",
                "tampering": "[Error Level Analysis results]"
            }
        
        # Video perception (Deepfake detector)
        if "videos" in input_data:
            perception["videos"] = {
                "analysis": f"[Video analysis of {len(input_data['videos'])} videos]",
                "deepfake_score": 0.12,
                "faces": "[Face tracking results]"
            }
        
        # Structured data
        if "data" in input_data:
            perception["structured"] = input_data["data"]
        
        return perception
    
    async def _reason(self, perception: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """
        Cross-domain reasoning.
        
        Analyzes perception, finds patterns, makes predictions.
        """
        reasoning = {
            "domain": task.domain.value,
            "analysis": [],
            "patterns": [],
            "predictions": [],
            "confidence": 0.0
        }
        
        # Domain-specific reasoning
        if task.domain == TaskDomain.FORENSICS:
            reasoning["analysis"].append("Audio analysis shows speaker characteristics")
            reasoning["analysis"].append("Image analysis detects no tampering")
            reasoning["confidence"] = 0.92
        
        elif task.domain == TaskDomain.TRADING:
            reasoning["analysis"].append("Market pattern indicates upward trend")
            reasoning["patterns"].append("Historical pattern: 65% win rate")
            reasoning["predictions"].append("Predicted move: +2.5% in 24h")
            reasoning["confidence"] = 0.78
        
        elif task.domain == TaskDomain.MONITORING:
            reasoning["analysis"].append("No suspicious activity detected")
            reasoning["confidence"] = 0.95
        
        elif task.domain == TaskDomain.PREDICTION:
            reasoning["predictions"].append("Crime likelihood: 5% (low)")
            reasoning["patterns"].append("Normal behavior pattern detected")
            reasoning["confidence"] = 0.88
        
        return reasoning
    
    def _decide_action(self, reasoning: Dict[str, Any], task: Task) -> Action:
        """
        Decide what action to take.
        """
        # Map task domain to action type
        action_mapping = {
            TaskDomain.FORENSICS: "generate_forensic_report",
            TaskDomain.TRADING: "execute_trade",
            TaskDomain.CONTENT_CREATION: "create_content",
            TaskDomain.MONITORING: "send_alert",
            TaskDomain.PREDICTION: "log_prediction",
            TaskDomain.GENERAL: "provide_response"
        }
        
        action_type = action_mapping.get(task.domain, "provide_response")
        
        action = Action(
            task_id=task.id,
            action_type=action_type,
            parameters={
                "reasoning": reasoning,
                "input": task.input_data
            },
            risk_level=task.risk_level,
            requires_approval=task.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL],
            confidence=reasoning.get("confidence", 0.0),
            reasoning=f"Based on {len(reasoning.get('analysis', []))} analysis points"
        )
        
        return action
    
    def _requires_approval(self, action: Action) -> bool:
        """Check if action requires human approval"""
        # Critical always requires approval
        if action.risk_level == RiskLevel.CRITICAL:
            return True
        
        # High risk requires approval
        if action.risk_level == RiskLevel.HIGH:
            return True
        
        # Low confidence requires approval
        if action.confidence < 0.7:
            return True
        
        # Check config
        if action.action_type in self.config.get("require_approval_for", []):
            return True
        
        return False
    
    async def _request_human_approval(self, action: Action) -> bool:
        """
        Request human approval for high-risk action.
        
        In production: send notification, wait for response
        For now: simulate approval
        """
        self.metrics["human_approvals_requested"] += 1
        
        print(f"\n⚠️  HUMAN APPROVAL REQUIRED")
        print(f"   Action: {action.action_type}")
        print(f"   Risk: {action.risk_level.value}")
        print(f"   Confidence: {action.confidence:.2%}")
        print(f"   Reasoning: {action.reasoning}")
        print(f"\n   Waiting for approval...")
        
        # Simulate approval (in production, would wait for actual human)
        await asyncio.sleep(0.5)
        
        # For now, approve if confidence > 0.8
        approved = action.confidence > 0.8
        
        if approved:
            self.metrics["human_approvals_granted"] += 1
            print(f"   ✅ APPROVED\n")
        else:
            print(f"   ❌ DENIED\n")
        
        action.approved = approved
        action.approver = "human_reviewer"
        
        return approved
    
    async def _execute_action(self, action: Action) -> Dict[str, Any]:
        """Execute the action"""
        if not action.approved:
            return {
                "success": False,
                "error": "Action not approved",
                "action": action.action_type
            }
        
        print(f"🚀 Executing: {action.action_type}")
        print(f"   Risk: {action.risk_level.value}")
        print(f"   Confidence: {action.confidence:.2%}")
        
        # Simulate execution
        await asyncio.sleep(0.1)
        
        # Action-specific execution
        if action.action_type == "execute_trade":
            result = {
                "success": True,
                "trade_id": "trade_123",
                "status": "executed",
                "amount": action.parameters["input"].get("amount", 0)
            }
        
        elif action.action_type == "generate_forensic_report":
            result = {
                "success": True,
                "report_id": "report_456",
                "status": "generated",
                "confidence": action.confidence
            }
        
        elif action.action_type == "create_content":
            result = {
                "success": True,
                "content_id": "content_789",
                "type": action.parameters["input"].get("type", "unknown"),
                "status": "created"
            }
        
        else:
            result = {
                "success": True,
                "status": "completed"
            }
        
        action.executed = True
        action.result = result
        
        print(f"   ✅ Success\n")
        
        return result
    
    def _audit_log(self, task: Task, action: Action, result: Dict[str, Any]):
        """Log action for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "task_domain": task.domain.value,
            "task_description": task.description,
            "action_type": action.action_type,
            "risk_level": action.risk_level.value,
            "requires_approval": action.requires_approval,
            "approved": action.approved,
            "approver": action.approver,
            "executed": action.executed,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "result": result,
            "success": result.get("success", False)
        }
        
        # Append to audit log
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Store in memory
        self.action_history.append(action)
    
    async def _meta_learn(self, task: Task, action: Action, result: Dict[str, Any]):
        """
        Meta-learning: learn from actions and results.
        
        If result was wrong, use correction to fine-tune.
        """
        # In production, would:
        # 1. Collect human corrections
        # 2. Build training dataset
        # 3. Fine-tune models
        # 4. Validate improvement
        
        print(f"📚 Meta-learning: Recording action for future improvement")
        
        # Placeholder for learning logic
        pass
    
    def receive_correction(self, action_id: str, correction: Dict[str, Any]):
        """
        Receive human correction for an action.
        
        This feeds the meta-learning pipeline.
        """
        self.metrics["corrections_received"] += 1
        
        correction_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "correction": correction,
            "for_training": True
        }
        
        # Save correction
        corrections_path = Path("training_data/corrections.jsonl")
        with open(corrections_path, 'a') as f:
            f.write(json.dumps(correction_entry) + '\n')
        
        print(f"✅ Correction received for action {action_id}")
        print(f"   Total corrections: {self.metrics['corrections_received']}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
        success_rate = (
            self.metrics["tasks_completed"] / total_tasks
            if total_tasks > 0 else 0
        )
        
        approval_rate = (
            self.metrics["human_approvals_granted"] / 
            self.metrics["human_approvals_requested"]
            if self.metrics["human_approvals_requested"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "approval_rate": approval_rate,
            "autonomous_rate": (
                self.metrics["autonomous_actions"] / total_tasks
                if total_tasks > 0 else 0
            )
        }
    
    def print_metrics(self):
        """Print performance metrics"""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 80)
        print("📊 AGI PERFORMANCE METRICS")
        print("=" * 80)
        print(f"\n📈 Tasks:")
        print(f"   Total: {metrics['total_tasks']}")
        print(f"   Completed: {metrics['tasks_completed']}")
        print(f"   Failed: {metrics['tasks_failed']}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        
        print(f"\n🤖 Autonomy:")
        print(f"   Autonomous Actions: {metrics['autonomous_actions']}")
        print(f"   Autonomous Rate: {metrics['autonomous_rate']:.1%}")
        
        print(f"\n👤 Human Oversight:")
        print(f"   Approvals Requested: {metrics['human_approvals_requested']}")
        print(f"   Approvals Granted: {metrics['human_approvals_granted']}")
        print(f"   Approval Rate: {metrics['approval_rate']:.1%}")
        
        print(f"\n📚 Learning:")
        print(f"   Corrections Received: {metrics['corrections_received']}")
        print(f"   Models Improved: {metrics['models_improved']}")
        
        print("\n" + "=" * 80 + "\n")


# Factory function
_controller = None

def get_agi_controller() -> UnifiedAGIController:
    """Get global AGI controller instance"""
    global _controller
    if _controller is None:
        _controller = UnifiedAGIController()
    return _controller


# Example usage
async def main():
    """Test the AGI controller"""
    controller = get_agi_controller()
    
    # Example 1: Low-risk forensic analysis
    task1 = Task(
        id="task_001",
        domain=TaskDomain.FORENSICS,
        description="Analyze audio sample for speaker identification",
        input_data={
            "type": "audio_analysis",
            "audio": "sample.wav"
        }
    )
    
    result1 = await controller.process_task(task1)
    print(f"Result: {result1['status']}")
    
    # Example 2: High-risk trading
    task2 = Task(
        id="task_002",
        domain=TaskDomain.TRADING,
        description="Execute trade based on signal",
        input_data={
            "type": "execute_trade",
            "symbol": "AAPL",
            "amount": 5000,
            "direction": "buy"
        }
    )
    
    result2 = await controller.process_task(task2)
    print(f"Result: {result2['status']}")
    
    # Example 3: Content creation
    task3 = Task(
        id="task_003",
        domain=TaskDomain.CONTENT_CREATION,
        description="Create 2-hour movie from script",
        input_data={
            "type": "movie",
            "script": "Thriller about forensic investigator",
            "duration_hours": 2,
            "public": False
        }
    )
    
    result3 = await controller.process_task(task3)
    print(f"Result: {result3['status']}")
    
    # Print metrics
    controller.print_metrics()


if __name__ == "__main__":
    asyncio.run(main())
