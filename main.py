from src.agents import (
    prepare_experiment,
    save_agent,
    load_saved_experiment,
    LossesCallback,
)
from src.utils import load_json_params, generate_param_combinations

DATASETS = ["chinatown", "ecg200"]

parameters = load_json_params("./params/agents/cfe_rl.json")
combinations = generate_param_combinations(parameters)

for dataset in DATASETS:
    for combo in combinations:
        hash_experiment, data, environment, agent = prepare_experiment(
            dataset=dataset, params=combo, dataset_path="/UCR/"
        )

        timesteps = combo.get("timesteps", 100_000)
        test_data, test_labels, test_nuns = data.get_test_samples()

        losses_callback = LossesCallback(
            total_timesteps=timesteps,
            tensorboard_path=f"./results/{hash_experiment}",
            model=data.model,
            samples=test_data,
            labels=test_labels,
            nuns=test_nuns,
            env=environment,
        )

        agent.learn(
            total_timesteps=timesteps,
            callback=losses_callback,
            progress_bar=True,
        )
        save_agent(hash_experiment, data, agent, environment)

# agent2, env2, data2 = load_saved_experiment(hash_experiment)
