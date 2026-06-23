import sys
sys.path.insert(0, '.')
from src.runner.config_loader import load_config
from src.runner.runner import ExperimentRunner

config = load_config('experiments/configs/defense_factorial.yaml')
config.runs_per_condition = 10
config.results_path = 'results/qwq_ollama_version_test/results.jsonl'
config.models = [m for m in config.models if 'qwq' in m.get('model_name','')]
config.defenses = [d for d in config.defenses if d.get('name') == 'no_defense']
config.attacks = [a for a in config.attacks if a.get('type') == 'delayed_trigger']
runner = ExperimentRunner(config)
runner.run_all()
