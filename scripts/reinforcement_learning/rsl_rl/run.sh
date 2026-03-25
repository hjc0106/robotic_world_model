# wmp
python scripts/reinforcement_learning/rsl_rl/train.py  \
    --task MBRL-Velocity-Rough-Unitree-A1-WMP-v0  \
    --enable_camera \
    --headless \
    --num_envs=4

python scripts/reinforcement_learning/rsl_rl/wmp_play.py  \
    --task MBRL-Velocity-Rough-Unitree-A1-WMP-Play-v0  \
    --enable_camera \
    --checkpoint /home/qylab/hjc_space/codes/robot_lab/robotic_world_model/logs/rsl_rl/unitree_a1_rough_wmp_ppo/model_61300.pt \
    --terrain stair

# pretrain
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --num_envs 16 \
    --headless
# finetune
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --headless \
    --load_run 2026-03-18_10-12-57_pretrain \
    --system_dynamics_load_path logs/rsl_rl/anymal_d_flat/2026-03-18_10-12-57_pretrain/model_3999.pt
# visualize
python scripts/reinforcement_learning/rsl_rl/visualize.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
    --checkpoint logs/rsl_rl/anymal_d_flat/2026-03-18_10-12-57_finetune/model_3999.pt \
    --system_dynamics_load_path logs/rsl_rl/anymal_d_flat/2026-03-18_10-12-57_finetune/model_3999.pt 