python experiment.py \
    --algorithm.name=uni_ppo.ppo \
    --environment.name="multi_robot" \
    --runner.track_console=True \
    --runner.load_model=pre_trained_model \
    --algorithm.determine_fastest_cpu_for_gpu=False \
    --algorithm.evaluation_frequency=-1 \
    --runner.mode=test \
    --environment.mode=test \
    --environment.add_goal_arrow=True \
    --environment.nr_envs=17 \
    --environment.multi_render=True \
    --environment.render=False

# python experiment.py \
#     --algorithm.name=uni_ppo.ppo \
#     --environment.name="cosmo" \
#     --runner.track_console=True \
#     --runner.load_model=pre_trained_model \
#     --algorithm.determine_fastest_cpu_for_gpu=False \
#     --runner.mode=test \
#     --environment.mode=test \
#     --environment.add_goal_arrow=True \
#     --environment.nr_envs=1 \
#     --environment.multi_render=False \
#     --environment.render=False