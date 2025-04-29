export CUDA_VISIBLE_DEVICES=2

# debugging
# python train.py \
#     --model_type=est_coord \
#     --device=cuda \
#     --exp_name=debug \
#     --max_iter=100


# default
# python train.py \
#     --model_type=est_coord \
#     --device=cuda \
#     --exp_name=coord_default \
#     --max_iter=10000

python train.py \
    --model_type=est_coord \
    --device=cuda \
    --exp_name=coord_default \
    --checkpoint=exps/coord_default/checkpoint/checkpoint_10000.pth \
    --max_iter=50000

