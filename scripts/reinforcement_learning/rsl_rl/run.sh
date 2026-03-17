CUDA_VISIBLE_DEVICES=4 python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=MBRL-Velocity-Rough-Unitree-A1-WMP-v0 \
    --num_envs=1024 \
    --headless \
    --video \
    --max_iterations=20000 \
    --enable_camera \
    --experiment_name=unitree_a1_rough_wmp_ppo_stand_new_amp \
    --resume True \
    --load_run 2026-03-16_12-37-17

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless \
    --distributed 
