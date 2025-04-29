export CUDA_VISIBLE_DEVICES=2

CKPT=exps/pose_t1r3/checkpoint/checkpoint_40000.pth

python test.py \
    --checkpoint=$CKPT \
    --mode=val \
    --device=cuda