
export resume_path="./ckpt/safe_stage_0.pth"

# export input_num=512
export input_num=380

# export part_pred_npoints=256
export part_pred_npoints=128

export cuda_ids=0,1,2,3,4,5,6,7
export num_gpus=8

export recon_prior=6
# export recon_prior=9

export sel_mode_trans=18

CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus} run_unsup_arti_align.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=${input_num} --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=${part_pred_npoints} --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='hoi4d' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="safe" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=${recon_prior} --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="motion_safe" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path=${resume_path} --mtx-based-axis-regression=False  --sel-mode-trans=${sel_mode_trans}
