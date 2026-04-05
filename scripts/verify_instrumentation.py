#!/usr/bin/env python3
"""Verify that mechanistic instrumentation is working correctly.

This script runs a small pilot (10 runs) to verify:
1. Mechanistic tags are populated
2. Defense logs capture modifications
3. RAG logs show retrieved vs. filtered documents
4. Agent logs capture tool calls
5. Confound detection logic works

Usage:
    .venv/bin/python scripts/verify_instrumentation.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner.runner import ExperimentRunner
from src.runner.config_loader import load_config


def main():
    print("=" * 80)
    print("INSTRUMENTATION VERIFICATION PILOT")
    print("=" * 80)
    print()
    
    # Load config
    config_path = Path("experiments/configs/default_factorial.yaml")
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    
    print(f"📋 Loading config: {config_path}")
    config = load_config(str(config_path))
    
    # Reduce to 10 runs for verification
    config.runs_per_condition = 1
    config.results_path = "results/verify_instrumentation/pilot_results.json"
    
    # Select 1 attack (delayed_trigger), 2 defenses, 1 model for quick verification
    config.attacks = [a for a in config.attacks if a.get("type") == "delayed_trigger"][:1]
    config.models = config.models[:1]    # Just qwen3:8b
    config.defenses = [d for d in config.defenses if d.get("type") in ["none", "minimizer_only", "rag_sanitizer"]][:3]
    
    print(f"✓ Config loaded")
    print(f"  Attacks: {len(config.attacks)}")
    print(f"  Defenses: {len(config.defenses)}")
    print(f"  Models: {len(config.models)}")
    print(f"  Runs per condition: {config.runs_per_condition}")
    print()
    
    # Run experiment
    print("🚀 Running verification pilot...")
    runner = ExperimentRunner(config)
    results = runner.run_all(results_path=config.results_path)
    
    print(f"✓ Pilot complete: {len(results)} runs")
    print()
    
    # Analyze results
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    checks = {
        "mechanistic_tags_populated": 0,
        "defense_logs_populated": 0,
        "rag_logs_populated": 0,
        "agent_logs_populated": 0,
        "mechanism_detected": 0,
        "confound_risk_assigned": 0,
    }
    
    for i, result in enumerate(results):
        print(f"Run {i+1}/{len(results)}: {result.condition['attack']['type']} + {result.condition['defense']['type']}")
        
        # Check mechanistic_tags
        if result.mechanistic_tags:
            checks["mechanistic_tags_populated"] += 1
            mechanism = result.mechanistic_tags.get("mechanism", "unknown")
            confound_risk = result.mechanistic_tags.get("confound_risk", "unknown")
            print(f"  ✓ mechanistic_tags: mechanism={mechanism}, risk={confound_risk}")
            
            if mechanism != "unknown":
                checks["mechanism_detected"] += 1
            if confound_risk != "unknown":
                checks["confound_risk_assigned"] += 1
        else:
            print(f"  ✗ mechanistic_tags: None")
        
        # Check defense_logs
        if result.defense_logs:
            checks["defense_logs_populated"] += 1
            print(f"  ✓ defense_logs: {len(result.defense_logs)} entries")
        else:
            print(f"  ✗ defense_logs: empty")
        
        # Check rag_logs
        if result.rag_logs:
            checks["rag_logs_populated"] += 1
            print(f"  ✓ rag_logs: {len(result.rag_logs)} entries")
        else:
            print(f"  ✗ rag_logs: empty (expected for no_attack)")
        
        # Check agent_logs
        if result.agent_logs:
            checks["agent_logs_populated"] += 1
            print(f"  ✓ agent_logs: {len(result.agent_logs)} entries")
        else:
            print(f"  ✗ agent_logs: empty")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_runs = len(results)
    for check, count in checks.items():
        pct = (count / total_runs * 100) if total_runs > 0 else 0
        status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "✗"
        print(f"{status} {check}: {count}/{total_runs} ({pct:.0f}%)")
    
    print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    all_good = all(count >= total_runs * 0.8 for count in checks.values())
    
    if all_good:
        print("✅ Instrumentation is working correctly!")
        print()
        print("You can now run the full factorial experiment:")
        print("  .venv/bin/python scripts/pilot_experiment.py experiments/configs/default_factorial.yaml")
        print()
        print("Or run the DTA ablation:")
        print("  .venv/bin/python scripts/pilot_experiment.py experiments/configs/dta_ablation.yaml")
        return 0
    else:
        print("⚠️  Some instrumentation checks failed. Review the results above.")
        print()
        print("Common issues:")
        print("  - mechanistic_tags not populated: Check compute_mechanistic_tags() in runner")
        print("  - defense_logs not populated: Check defense.apply() returns DefenseLog")
        print("  - rag_logs not populated: Check RAGTool.query() logs retrieved docs")
        print("  - agent_logs not populated: Check agent.run_session() captures tool calls")
        return 1


if __name__ == "__main__":
    sys.exit(main())
