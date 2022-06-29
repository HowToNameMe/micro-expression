### ---------------------------training-------------------------------
# mask warp 
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/Mixed_data-10-8-wMaskWarp-aug.yaml 

### ---------------------------generation-------------------------------
## mask warp
# relative nobest
CUDA_VISIBLE_DEVICES=0 python demo.py --config config/Mixed_data-10-8-wMaskWarp-aug.yaml  \
    --checkpoint 'path to the checkpoints' \
    --result_video './ckpt/relative-nobest/wMaskWarp' \
    --mode 'relative'

