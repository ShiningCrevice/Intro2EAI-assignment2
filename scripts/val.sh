export CUDA_VISIBLE_DEVICES=2

CKPT_POSE=exps/pose_t1r3/checkpoint/checkpoint_40000.pth
CKPT_COORD=exps/coord_default/checkpoint/checkpoint_50000.pth

python eval.py \
    --checkpoint=$CKPT_POSE \
    --mode=val \
    --device=cuda \
    --vis=0 \
    --headless=1

python eval.py \
    --checkpoint=$CKPT_COORD \
    --mode=val \
    --device=cuda \
    --vis=0 \
    --headless=1