

export resume_path="./ckpt/eyeglasses_stage_0.pth"

export input_num=480

export part_pred_npoints=100
export num_iters=1


export cuda_ids=0,1,2,3,4,5,6,7
export num_gpus=8



#### run mode use-equi=35 stage = 1 thulab ####
CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus} run_unsup_arti_align.py experiment -d "./data" --init-lr=1e-4 --num-iters=${num_iters} --global-rot=1 --input-num=${input_num} --bsz=1 --nmasks=3 --use-equi=35 --part-pred-npoints=${part_pred_npoints} --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --kanchor=60 --cent-trans=3 --shape-type="eyeglasses" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="motion_eyeglasses" --kpconv-kanchor=60  --sel-mode=29 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=1 --resume-path=${resume_path}  --pred-axis=1  --mtx-based-axis-regression=True

