"""
Command Line Interface for the Council of Infinite Innovators.
"""

import asyncio
import sys
from typing import List, Optional
from pathlib import Path
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the project root to Python path for imports
sys.path.append('.')

app = typer.Typer(
    name="council",
    help="🏛️ Council of Infinite Innovators - AI Agent Framework",
    no_args_is_help=True
)

console = Console()

@app.command()
def run(
    agent: str = typer.Option(..., "--agent", "-a", help="Agent to consult (strategist, architect, engineer, etc.)"),
    input: str = typer.Option(..., "--input", "-i", help="Question or problem to address"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override default LLM model"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override default LLM provider"),
):
    """Run a single agent consultation."""
    
    async def _run():
        try:
            from council import build_council, AGENT_TYPES
            
            if agent not in AGENT_TYPES:
                available = ", ".join(AGENT_TYPES.keys())
                print(f"❌ Unknown agent: {agent}")
                print(f"Available agents: {available}")
                return
            
            # Override settings if provided
            if model or provider:
                from council.config import SETTINGS
                if model:
                    SETTINGS.model = model
                if provider:
                    SETTINGS.provider = provider
            
            # Build council with single agent
            graph = await build_council([agent])
            
            print(f"\n🤔 Consulting [bold blue]{agent}[/bold blue]...")
            print(f"📝 Question: [italic]{input}[/italic]\n")
            
            # Run consultation
            response = await graph.run_single_agent(input, agent)
            
            # Display response in a panel
            panel = Panel(
                response,
                title=f"💭 {agent.title()} Perspective",
                border_style="blue",
                padding=(1, 2)
            )
            print(panel)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()
    
    asyncio.run(_run())

@app.command()
def council(
    agents: str = typer.Option(..., "--agents", "-a", help="Comma-separated list of agents to consult"),
    input: str = typer.Option(..., "--input", "-i", help="Question or problem to address"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override default LLM model"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override default LLM provider"),
):
    """Run a full council consultation with multiple agents."""
    
    async def _run():
        try:
            from council import build_council, AGENT_TYPES
            
            # Parse agent list
            agent_list = [a.strip() for a in agents.split(",")]
            
            # Validate agents
            unknown_agents = [a for a in agent_list if a not in AGENT_TYPES]
            if unknown_agents:
                available = ", ".join(AGENT_TYPES.keys())
                print(f"❌ Unknown agents: {unknown_agents}")
                print(f"Available agents: {available}")
                return
            
            # Override settings if provided
            if model or provider:
                from council.config import SETTINGS
                if model:
                    SETTINGS.model = model
                if provider:
                    SETTINGS.provider = provider
            
            # Build council
            graph = await build_council(agent_list)
            
            print(f"\n🏛️ Convening Council of {len(agent_list)} agents...")
            print(f"👥 Agents: [bold blue]{', '.join(agent_list)}[/bold blue]")
            print(f"📝 Question: [italic]{input}[/italic]\n")
            
            # Run council session
            response = await graph.run(input, agent_list)
            
            # Display response
            panel = Panel(
                response,
                title="🏛️ Council Synthesis",
                border_style="green",
                padding=(1, 2)
            )
            print(panel)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()
    
    asyncio.run(_run())

@app.command()
def list_agents():
    """List all available agents."""
    from council import AGENT_TYPES
    
    table = Table(title="🏛️ Council of Infinite Innovators - Available Agents")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Role", style="magenta")
    
    roles = {
        "strategist": "Market intelligence and strategic positioning",
        "architect": "System design and scalability",
        "engineer": "Implementation and production readiness",
        "designer": "User experience and interface design",
        "entrepreneur": "Business opportunities and market validation",
        "futurist": "Trend analysis and future scenarios",
        "economist": "Financial modeling and economic analysis",
        "ethicist": "Responsible AI and ethical considerations",
        "philosopher": "Fundamental assumptions and deeper implications",
        "cultural_translator": "Cross-cultural adaptation and localization",
    }
    
    for agent_name in sorted(AGENT_TYPES.keys()):
        role = roles.get(agent_name, "Specialized AI agent")
        table.add_row(agent_name, role)
    
    console.print(table)

@app.command()
def agentic(
    agent: str = typer.Option("strategist", help="Agent to run in agentic mode"),
    input: str = typer.Option(..., help="Task for the agent"),
):
    """
    Run an agent in AGENTIC mode with tool use and iterative reasoning.
    
    Agentic agents can:
    - Use tools (web search, code execution, calculations)
    - Reason iteratively (think → act → observe → repeat)
    - Work autonomously toward goals
    
    Example:
        python -m cli.app agentic --agent strategist --input "Research AI trends and calculate market size"
    """
    from council.agents.agentic import AgenticStrategist, AgenticEngineer, AgenticResearcher
    from council.llm import get_llm
    from council.agents.base import Message
    
    # Map agent names to agentic classes
    agentic_agents = {
        "strategist": AgenticStrategist,
        "engineer": AgenticEngineer,
        "researcher": AgenticResearcher,
        "futurist": AgenticResearcher,
    }
    
    if agent not in agentic_agents:
        console.print(f"[red]❌ Agent '{agent}' not available in agentic mode[/red]")
        console.print(f"[yellow]Available: {', '.join(agentic_agents.keys())}[/yellow]")
        return
    
    console.print(f"\n🤖 Running [cyan]{agent}[/cyan] in AGENTIC mode...")
    console.print(f"📝 Task: {input}\n")
    
    async def run_agentic():
        llm = get_llm()
        agent_class = agentic_agents[agent]
        agentic_agent = agent_class(llm)
        
        messages = [Message(role="user", content=input)]
        
        with console.status(f"[bold green]Agent is thinking and using tools..."):
            result = await agentic_agent.run(messages)
        
        panel = Panel(
            result,
            title=f"🤖 {agent.title()} (Agentic Mode)",
            border_style="cyan"
        )
        console.print(panel)
    
    asyncio.run(run_agentic())


@app.command()
def ensemble(
    input: str = typer.Option(..., help="Query for the model ensemble"),
    models: int = typer.Option(3, help="Number of models to use"),
    strategy: str = typer.Option("best_of_n", help="Ensemble strategy: best_of_n, weighted, consensus"),
    task_type: str = typer.Option(None, help="Task type: code, reasoning, creative, general"),
):
    """
    Query MULTIPLE top models and combine their responses.
    
    Uses GPT-4, Claude, Gemini, Llama, etc. and intelligently combines outputs.
    This is better than any single model!
    
    Example:
        python -m cli.app ensemble --input "Explain quantum computing" --models 3
    """
    from council.model_ensemble import get_ensemble, EnsembleStrategy
    
    console.print(f"\n🎭 Querying {models} top models...")
    console.print(f"📝 Question: {input}")
    console.print(f"🎯 Strategy: {strategy}\n")
    
    async def run_ensemble():
        ensemble = get_ensemble()
        
        messages = [{"role": "user", "content": input}]
        
        # Map strategy string to enum
        strategy_map = {
            "best_of_n": EnsembleStrategy.BEST_OF_N,
            "weighted": EnsembleStrategy.WEIGHTED,
            "consensus": EnsembleStrategy.CONSENSUS,
        }
        
        with console.status("[bold green]Models are thinking..."):
            result = await ensemble.ensemble_query(
                messages=messages,
                strategy=strategy_map.get(strategy, EnsembleStrategy.BEST_OF_N),
                num_models=models,
                task_type=task_type,
            )
        
        # Display result
        panel = Panel(
            result["answer"],
            title=f"🎭 Ensemble Response (Selected: {result.get('selected_model', 'N/A')})",
            border_style="magenta"
        )
        console.print(panel)
        
        # Show metadata
        console.print(f"\n📊 Metadata:")
        console.print(f"  Confidence: {result['confidence']:.0%}")
        console.print(f"  Models used: {result.get('total_models', models)}")
        console.print(f"  Avg latency: {result.get('avg_latency', 0):.2f}s")
        console.print(f"  Total cost: ${result.get('total_cost', 0):.4f}")
        
        if "all_responses" in result:
            console.print(f"\n🔍 All model responses:")
            for resp in result["all_responses"]:
                console.print(f"  • {resp['model']}: {resp['preview']}")
    
    asyncio.run(run_ensemble())


@app.command()
def learning_stats():
    """Show continuous learning statistics."""
    from council.continuous_learning import get_learner
    
    learner = get_learner()
    stats = learner.get_learning_stats()
    
    console.print("\n📚 Continuous Learning Statistics\n")
    
    if "message" in stats:
        console.print(f"[yellow]{stats['message']}[/yellow]")
        console.print("\n💡 Start using ensemble mode to collect training data!")
        return
    
    table = Table(title="Training Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Examples", str(stats["total_examples"]))
    table.add_row("High Quality (>0.8)", str(stats["high_quality_examples"]))
    table.add_row("Ready for Fine-tuning", "✅ Yes" if stats["ready_for_finetuning"] else f"❌ No (need {stats['examples_needed']} more)")
    table.add_row("Average Quality", f"{stats['avg_quality']:.2%}")
    table.add_row("Examples with Feedback", str(stats["examples_with_feedback"]))
    
    console.print(table)
    
    # Task distribution
    if stats["task_distribution"]:
        console.print("\n📊 Task Distribution:")
        for task, count in sorted(stats["task_distribution"].items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {task}: {count} examples")


@app.command()
def finetune(
    provider: str = typer.Option("huggingface", help="Provider: openai or huggingface"),
    model: str = typer.Option(None, help="Base model to fine-tune"),
):
    """
    Create YOUR OWN custom model from collected training data.
    
    OPTION A (OpenAI): Fast, vendor lock-in, $10-50/month usage
    OPTION B (HuggingFace): Full control, YOU OWN IT, $0 if self-hosted
    
    For HuggingFace (Option B):
    - Local GPU: python scripts/finetune_hf_model.py
    - Google Colab (FREE GPU): See COLAB_FINETUNING.md
    
    Example:
        python -m cli.app finetune --provider openai
        python -m cli.app finetune --provider huggingface --model meta-llama/Llama-3.2-3B-Instruct
    """
    from council.continuous_learning import get_learner
    
    learner = get_learner()
    stats = learner.get_learning_stats()
    
    console.print("\n🎓 Fine-tuning Custom Model\n")
    
    if not stats.get("ready_for_finetuning"):
        console.print(f"[red]❌ Not enough training data yet[/red]")
        console.print(f"[yellow]Need {stats.get('examples_needed', 100)} more high-quality examples[/yellow]")
        console.print(f"\n💡 Use ensemble mode to collect more data:")
        console.print(f"   python -m cli.app ensemble --input 'your query' --models 3")
        return
    
    if provider == "huggingface":
        console.print("\n[cyan]📋 Option B: HuggingFace Fine-Tuning (RECOMMENDED)[/cyan]")
        console.print("   ✅ YOU OWN THE MODEL")
        console.print("   ✅ 100% ownership and control")
        console.print("   ✅ $0 training (Google Colab FREE GPU)")
        console.print("   ✅ $0 usage (self-hosted)")
        console.print("   ✅ Can run offline, no vendor lock-in")
        console.print("\n🚀 To fine-tune:")
        console.print("   1. Local GPU: python scripts/finetune_hf_model.py")
        console.print("   2. Google Colab (FREE): See COLAB_FINETUNING.md")
        console.print("\n📚 Full guide: OPTION_B_HUGGINGFACE.md")
        return
    
    async def run_finetune():
        if provider == "openai":
            console.print("[cyan]� Fine-tuning with OpenAI...[/cyan]")
            console.print("   ⚠️  Note: OpenAI owns base model, ongoing costs")
            result = await learner.finetune_openai(model or "gpt-3.5-turbo")
        else:
            console.print(f"[red]Unknown provider: {provider}[/red]")
            return
        
        if "error" in result:
            console.print(f"[red]❌ {result['error']}[/red]")
        else:
            console.print(f"[green]✅ Fine-tuning initiated![/green]")
            for key, value in result.items():
                console.print(f"  {key}: {value}")
    
    asyncio.run(run_finetune())


@app.command()
def models():
    """Show today's model rotation assignments."""
    from council.model_rotation import print_daily_assignments
    
    print()
    print_daily_assignments()
    print()

@app.command()
def validate():
    """Validate the council configuration and prompts."""
    from council.prompts.loaders import validate_prompts
    from council.config import SETTINGS
    
    print("🔍 Validating Council configuration...\n")
    
    # Check configuration
    try:
        SETTINGS.validate()
        print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # Check prompts
    print("\n📝 Validating prompts...")
    prompt_results = validate_prompts()
    
    for prompt_name, is_valid in prompt_results.items():
        status = "✅" if is_valid else "❌"
        print(f"{status} {prompt_name}")
    
    # Summary
    valid_count = sum(prompt_results.values())
    total_count = len(prompt_results)
    print(f"\n📊 Validation Summary: {valid_count}/{total_count} prompts valid")

@app.command()
def version():
    """Show version information."""
    from council import __version__
    print(f"Council of Infinite Innovators v{__version__}")

@app.command()
def forensic(
    input: str = typer.Option(..., help="Forensic evidence to analyze (logs, malware, network data)"),
    analysis_type: str = typer.Option(None, help="Type: log_analysis, malware_analysis, network_analysis, threat_intelligence"),
):
    """
    Forensic AI Agent - Analyze digital evidence and security incidents.
    
    Analyzes:
    - Security logs (system, application, firewall)
    - Malware samples and reports
    - Network traffic and packets
    - Threat intelligence
    
    Collects training data for YOUR forensic AI model!
    
    Example:
        python -m cli.app forensic --input "ERROR: Failed login from 192.168.1.100"
        python -m cli.app forensic --input "Malware detected: Trojan.Generic" --analysis-type malware_analysis
    """
    from council.agents.forensic import get_forensic_agent
    
    console.print("\n🔍 Forensic AI Agent\n", style="bold cyan")
    
    async def run_forensic():
        agent = get_forensic_agent()
        
        context = {}
        if analysis_type:
            context["analysis_type"] = analysis_type
        
        console.print(f"[yellow]Analyzing evidence...[/yellow]\n")
        
        result = await agent.analyze(input, context)
        
        # Display result
        panel = Panel(
            result.get("analysis", "No analysis available"),
            title=f"🔬 Forensic Analysis ({result.get('analysis_type', 'unknown')})",
            border_style="red"
        )
        console.print(panel)
        
        # Show IOCs
        if result.get("iocs_found"):
            console.print("\n🎯 Indicators of Compromise (IOCs):")
            for ioc_type, values in result["iocs_found"].items():
                console.print(f"  • {ioc_type}: {', '.join(values[:5])}")
                if len(values) > 5:
                    console.print(f"    ... and {len(values) - 5} more")
        
        # Show severity
        if "severity" in result:
            severity_color = {
                "Critical": "red",
                "High": "yellow",
                "Medium": "blue",
                "Low": "green"
            }.get(result["severity"], "white")
            console.print(f"\n⚠️  Severity: [{severity_color}]{result['severity']}[/{severity_color}]")
        
        console.print("\n💾 Forensic data saved for training YOUR forensic model!")
    
    asyncio.run(run_forensic())


@app.command()
def build():
    """
    Build YOUR unified AI models from collected training data.
    
    Builds:
    1. Unified Model - Handles everything (text, analysis, etc.)
    2. Forensic Model - Specialized for security/forensics
    
    Uses ALL collected data:
    - Ensemble interactions
    - Platform user data
    - Forensic analysis data
    
    Example:
        python -m cli.app build
    """
    import subprocess
    
    console.print("\n🏗️  Building YOUR AI Models\n", style="bold green")
    
    console.print("📊 This will:")
    console.print("   1. Collect ALL training data")
    console.print("   2. Prepare unified datasets")
    console.print("   3. Show training instructions")
    console.print()
    
    # Run build script
    result = subprocess.run(
        ["python", "scripts/build_unified_model.py"],
        cwd=Path.cwd(),
    )
    
    if result.returncode == 0:
        console.print("\n✅ Build process complete!")
        console.print("📖 Follow instructions above to fine-tune on Google Colab")
    else:
        console.print("\n❌ Build failed")


@app.command()
def unified(
    discover: bool = typer.Option(False, help="Discover all HuggingFace models"),
    stats: bool = typer.Option(False, help="Show platform statistics"),
):
    """
    Unified AI Platform - Use ALL HuggingFace models.
    
    Your users → interact with models → train YOUR unified model → profit!
    
    Examples:
        python -m cli.app unified --discover  # Find all HF models
        python -m cli.app unified --stats     # Show usage stats
    """
    from council.model_hub import get_hub, UnifiedModelTrainer
    
    console.print("\n🌐 Unified AI Platform\n", style="bold cyan")
    
    if discover:
        async def run_discovery():
            hub = get_hub()
            
            console.print("[yellow]🔍 Discovering ALL HuggingFace models...[/yellow]")
            console.print("[dim]This may take a few minutes...[/dim]\n")
            
            capabilities = await hub.discover_all_models(
                top_n_per_category=5,
                min_downloads=1000,
            )
            
            # Show summary
            console.print(f"\n[green]✅ Discovery complete![/green]\n")
            
            table = Table(title="Discovered Models")
            table.add_column("Capability", style="cyan")
            table.add_column("Models", style="green")
            table.add_column("Top Model", style="yellow")
            
            for cap, models in sorted(capabilities.items()):
                table.add_row(
                    cap,
                    str(len(models)),
                    models[0].model_id if models else "None"
                )
            
            console.print(table)
            
            total = sum(len(v) for v in capabilities.values())
            console.print(f"\n📊 Total: {total} models across {len(capabilities)} capabilities")
            console.print("\n💡 Start API: uvicorn api.unified:app --reload")
        
        asyncio.run(run_discovery())
    
    elif stats:
        hub = get_hub()
        trainer = UnifiedModelTrainer(hub)
        
        # Load cache
        if hub.load_cache():
            total_models = sum(len(v) for v in hub.capabilities.values())
            console.print(f"📦 Models available: {total_models}")
            console.print(f"🎯 Capabilities: {len(hub.capabilities)}\n")
        
        # Count interactions
        interaction_files = list(trainer.training_data_dir.glob("interactions_*.jsonl"))
        total_interactions = 0
        
        if interaction_files:
            console.print("📊 Training Data:")
            for file in interaction_files:
                with open(file) as f:
                    count = sum(1 for _ in f)
                    total_interactions += count
                    console.print(f"   {file.name}: {count:,} interactions")
            
            console.print(f"\n   Total: {total_interactions:,} interactions")
            console.print(f"   Ready for training: {'✅ Yes' if total_interactions >= 100 else '❌ No (need ' + str(100 - total_interactions) + ' more)'}")
        else:
            console.print("📊 No training data yet")
            console.print("   Start collecting: Use /task endpoint in API\n")
        
        console.print("\n💡 View full stats: GET http://localhost:8000/stats")
    
    else:
        console.print("Unified AI Platform - Use ALL HuggingFace models!\n")
        console.print("📚 Features:")
        console.print("   • Discovers ALL HF models (text, image, audio)")
        console.print("   • Auto-routes to best model for each task")
        console.print("   • Collects training data from user interactions")
        console.print("   • Trains YOUR unified model daily")
        console.print("   • Auto-updates with new models\n")
        
        console.print("🚀 Quick Start:")
        console.print("   1. python -m cli.app unified --discover")
        console.print("   2. uvicorn api.unified:app --reload")
        console.print("   3. Visit http://localhost:8000/docs")
        console.print("   4. Submit tasks via POST /task")
        console.print("   5. Check stats: python -m cli.app unified --stats\n")
        
        console.print("📖 Full guide: UNIFIED_PLATFORM.md")


@app.command()
def deepfake(
    media: str = typer.Option(..., help="Media URL, file path, or description to analyze"),
    media_type: str = typer.Option(None, help="Type: video, image, or audio"),
):
    """
    Deepfake Detection - Detect manipulated media (videos, images, audio).
    
    Detects:
    - Deepfake videos (face swaps, lip-sync manipulation)
    - AI-generated/manipulated images (synthetic faces)
    - Cloned voices and synthetic audio
    
    Collects training data for YOUR deepfake detection model!
    
    Example:
        python -m cli.app deepfake --media "suspicious_video.mp4"
        python -m cli.app deepfake --media "https://example.com/image.jpg" --media-type image
    """
    from council.agents.deepfake_detector import get_deepfake_detector
    
    console.print("\n🎭 Deepfake Detection AI\n", style="bold magenta")
    
    async def run_detection():
        detector = get_deepfake_detector()
        
        context = {}
        if media_type:
            context["media_type"] = media_type
        
        console.print(f"[yellow]Analyzing media for manipulation...[/yellow]\n")
        
        result = await detector.analyze(media, context)
        
        # Display result
        verdict_color = {
            "AUTHENTIC": "green",
            "LIKELY_AUTHENTIC": "blue",
            "SUSPICIOUS": "yellow",
            "LIKELY_DEEPFAKE": "orange",
            "DEEPFAKE": "red"
        }.get(result["verdict"], "white")
        
        panel = Panel(
            result.get("analysis", "No analysis available"),
            title=f"🎭 Deepfake Analysis ({result.get('media_type', 'unknown')})",
            border_style="magenta"
        )
        console.print(panel)
        
        # Show results
        console.print(f"\n🎯 Verdict: [{verdict_color}]{result['verdict']}[/{verdict_color}]")
        console.print(f"📊 Authenticity Score: {result['authenticity_score']:.0%}")
        
        # Show manipulation indicators
        if any(result.get("manipulation_indicators", {}).values()):
            console.print("\n⚠️  Manipulation Indicators:")
            for category, indicators in result["manipulation_indicators"].items():
                if indicators:
                    console.print(f"  • {category.replace('_', ' ').title()}: {', '.join(indicators[:3])}")
        
        # Show recommended models
        if result.get("recommended_models"):
            console.print(f"\n🤖 Recommended Models:")
            for model in result["recommended_models"][:3]:
                console.print(f"  • {model}")
        
        console.print("\n💾 Analysis saved for training YOUR deepfake detection model!")
    
    asyncio.run(run_detection())


@app.command()
def verify_document(
    document: str = typer.Option(..., help="Document image URL, file path, or description"),
    doc_type: str = typer.Option(None, help="Type: passport, id_card, drivers_license, certificate"),
):
    """
    Document Forgery Detection - Verify authenticity of IDs, passports, and documents.
    
    Detects forgery in:
    - Passports (security features, MRZ, holograms)
    - ID cards (national IDs, resident permits)
    - Driver's licenses
    - Certificates and diplomas
    - Bank statements
    - Legal documents
    
    Collects training data for YOUR document verification model!
    
    Example:
        python -m cli.app verify-document --document "passport_scan.jpg" --doc-type passport
        python -m cli.app verify-document --document "id_card.png"
    """
    from council.agents.document_forgery_detector import get_document_forgery_detector
    
    console.print("\n📄 Document Forgery Detection\n", style="bold blue")
    
    async def run_verification():
        detector = get_document_forgery_detector()
        
        context = {}
        if doc_type:
            context["document_type"] = doc_type
        
        console.print(f"[yellow]Analyzing document for forgery...[/yellow]\n")
        
        result = await detector.analyze(document, context)
        
        # Display result
        verdict_color = {
            "GENUINE": "green",
            "LIKELY_GENUINE": "blue",
            "SUSPICIOUS": "yellow",
            "LIKELY_FORGED": "orange",
            "FORGED": "red"
        }.get(result["verdict"], "white")
        
        panel = Panel(
            result.get("analysis", "No analysis available"),
            title=f"📄 Document Analysis ({result.get('document_type', 'unknown')})",
            border_style="blue"
        )
        console.print(panel)
        
        # Show results
        console.print(f"\n🎯 Verdict: [{verdict_color}]{result['verdict']}[/{verdict_color}]")
        console.print(f"📊 Authenticity Score: {result['authenticity_score']:.0%}")
        
        # Show forgery indicators
        if any(result.get("forgery_indicators", {}).values()):
            console.print("\n⚠️  Forgery Indicators:")
            for category, indicators in result["forgery_indicators"].items():
                if indicators:
                    console.print(f"  • {category.replace('_', ' ').title()}: {', '.join(indicators[:3])}")
        
        # Show extracted data
        if result.get("extracted_data"):
            console.print("\n📋 Extracted Data:")
            for key, value in result["extracted_data"].items():
                console.print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        # Show expected security features
        if result.get("security_features_expected"):
            console.print(f"\n🔒 Expected Security Features:")
            for feature in result["security_features_expected"][:5]:
                console.print(f"  • {feature}")
        
        console.print("\n💾 Analysis saved for training YOUR document verification model!")
    
    asyncio.run(run_verification())


if __name__ == "__main__":
    app()
