from src.agents import prepare_experiment, save_agent, load_saved_experiment
from src.utils import load_json_params, generate_param_combinations

DATASETS = ["chinatown", "ecg200"]

parameters = load_json_params("./params/agents/cfe_rl.json")
combinations = generate_param_combinations(parameters)

for dataset in DATASETS:
    for combo in combinations:
        hash_experiment, data, environment, agent = prepare_experiment(
            dataset, combo, dataset_path="/UCR/"
        )
        agent.learn(
            total_timesteps=combo.get("timesteps", 100_000),
            callback=None,
            progress_bar=True,
        )
        save_agent(hash_experiment, data, agent, environment)

# agent2, env2, data2 = load_saved_experiment(hash_experiment)
