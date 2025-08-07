#!/bin/bash
cd /home/andrea/git/UCLA/romela/RL-X
mamba activate urma
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch>=2.1.2" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12]"
cd /home/andrea/git/UCLA/romela/one_policy_to_run_them_all
pip install -e .

# Don't question this, only thing I found that works