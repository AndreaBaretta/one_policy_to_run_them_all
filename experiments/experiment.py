from rl_x.runner.runner import Runner
import one_policy_to_run_them_all.algorithms.uni_ppo
import one_policy_to_run_them_all.environments.multi_robot

if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "one_policy_to_run_them_all"])
    runner.run()