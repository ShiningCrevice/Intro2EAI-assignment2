export CUDA_VISIBLE_DEVICES=2

CKPT=exps/coord_default/checkpoint/checkpoint_50000.pth

python test.py \
    --checkpoint=$CKPT \
    --mode=val \
    --device=cuda