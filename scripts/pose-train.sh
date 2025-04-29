export CUDA_VISIBLE_DEVICES=2

# debugging
# python train.py \
#     --model_type=est_pose \
#     --device=cuda \
#     --exp_name=debug \
#     --max_iter=100

# python train.py \
#     --model_type=est_pose \
#     --device=cuda \
#     --exp_name=debug \
#     --max_iter=10100 \
#     --checkpoint=exps/default/checkpoint/checkpoint_10000.pth


# default
# python train.py \
#     --model_type=est_pose \
#     --device=cuda \
#     --exp_name=default

# python train.py \
#     --model_type=est_pose \
#     --device=cuda \
#     --exp_name=default \
#     --checkpoint=exps/default/checkpoint/checkpoint_10000.pth \
#     --max_iter=40000


# lambda_t, lambda_r = 0.5, 1.5
python train.py \
    --model_type=est_pose \
    --device=cuda \
    --exp_name=pose_t1r3 \
    --max_iter=40000 \
    --lambda_t=0.5 \
    --lambda_r=1.5 \