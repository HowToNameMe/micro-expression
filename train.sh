### ---------------------------training-------------------------------
# mask warp 
CUDA_VISIBLE_DEVICES=0,1 nohup python run.py --config config/Mixed_data-10-8-wMaskWarp.yaml \
1>Mixed_data-10-8-wMaskWarp.out 2>Mixed_data-10-8-wMaskWarp.err &

### ---------------------------generation-------------------------------
## mask warp
# relative nobest
CUDA_VISIBLE_DEVICES=0 python demo.py --config config/Mixed_data-10-8-wMaskWarp-aug.yaml  \
    --checkpoint 'log/Mixed_data-10-8-wMaskWarp-aug 25_06_22_16.11.40/00000049-checkpoint.pth.tar' \
    --result_video './ckpt/relative-nobest/wMaskWarp' \
    --mode 'relative'

