

export cuda_ids=0,1,2,3,4,5,6,7
export num_gpus=8


export cuda_ids=4
export num_gpus=1

export dataset_type="motion"
# export dataset_type="motion_partial"
export nmasks=3


# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus} run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=380 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=1 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv26" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1  --mtx-based-axis-regression=False --slot-single-cd=0 --glb-single-cd=0 --resume-path=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth 

# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus}  run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=512 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv19" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth 

# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus}   run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=512 --bsz=1 --nmasks=${nmasks} --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type=${dataset_type} --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv25" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=0 --permute-modes=1 --use-2d=0 --pred-axis=1    --mtx-based-axis-regression=False --slot-single-cd=0 --resume-path=/share/xueyi/ckpt/playground/model_20220428_13:42:19/ckpt/playground_net_Iter2800.pth 

# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus} run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=512 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv26" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1  --mtx-based-axis-regression=False --slot-single-cd=0 --glb-single-cd=0 --resume-path=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth  --add-normal-noise=0.02 --use-art-mode=True

# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch  --nproc_per_node=${num_gpus} run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=380 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv26" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1  --mtx-based-axis-regression=False --slot-single-cd=0 --glb-single-cd=0 --resume-path=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth --run-mode=eval --pre-compute-delta=1 --sel-mode-trans=28 --resume-path=/share/xueyi/ckpt/playground/model_20220424_11:18:15/ckpt/playground_net_Iter5000.pth  --pred-axis=1  --glb-single-cd=1 --slot-single-cd=1
# --mtx-based-axis-regression=True

# --resume-path=/share/xueyi/ckpt/playground/model_20220515_12:17:16/ckpt/playground_net_Iter400.pth  --rel-for-points=1 --mtx-based-axis-regression=False
# /share/xueyi/ckpt/playground/model_20220504_06:10:07/ckpt/playground_net_Iter1000.pth


## laptop evaluation ###
export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_partial_laptop"
export pre_compute_delta=0
export pre_compute_delta=1
export cuda_ids=7


CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=380 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=128 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion_partial' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="laptop" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=9 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv19" --kpconv-kanchor=60  --sel-mode=29 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=/share/xueyi/ckpt/playground/model_20220417_03:43:45/ckpt/playground_net_Iter800.pth    --run-mode=eval --pre-compute-delta=${pre_compute_delta} --sel-mode-trans=14 --resume-path=/share/xueyi/ckpt/playground/model_20220508_18:22:54/ckpt/playground_net_Iter600.pth --mtx-based-axis-regression=False --eval_data_sv_dict_fn=${eval_data_sv_dict_fn}


# ### oven evaluation ### ## motion oven --> arti_pose/motion_oven ###
# export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_oven"
# export pre_compute_delta=0
# export pre_compute_delta=2
# export cuda_ids=1

# # CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=512 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv19" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth   --run-mode=eval --pre-compute-delta=1 --sel-mode-trans=-1   --resume-path=/share/xueyi/ckpt/playground/model_20220515_12:17:16/ckpt/playground_net_Iter400.pth  --mtx-based-axis-regression=False

# export cuda_ids=4

# export resume_path=/share/xueyi/ckpt/playground/model_20220525_18:35:54/ckpt/playground_net_Iter800.pth
# # export resume_path=/share/xueyi/ckpt/playground/model_20220504_06:10:07/ckpt/playground_net_Iter1000.pth
# # export resume_path=/share/xueyi/ckpt/playground/model_20220503_06:48:34/ckpt/playground_net_Iter5600.pth
# # export resume_path=/share/xueyi/ckpt/playground/model_20220424_11:18:15/ckpt/playground_net_Iter5000.pth
# # export resume_path=/share/xueyi/ckpt/playground/model_20220425_00:58:49/ckpt/playground_net_Iter1800.pth
# # /share/xueyi/ckpt/playground/model_20220504_06:10:07/ckpt/playground_net_Iter1000.pth
# ## 

# CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=512 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="oven" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv26" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1  --mtx-based-axis-regression=False --slot-single-cd=0 --glb-single-cd=0  --sel-mode-trans=35 --run-mode=eval --pre-compute-delta=1 --resume-path=${resume_path}    --resume-path-glb=/share/xueyi/ckpt/playground/model_20220409_07:54:25/ckpt/playground_net_Iter9600.pth   # --add-normal-noise=0.02

# ### oven evaluation ### ## motion oven --> arti_pose/motion_oven ###

# 

## I do not like this checkpoint ##
# I do not #

#### ==== eyeglasses eval ==== ####

# /data1/sim/equi_arti_pose/iter_78.npy ##
# /data1/sim/equi_arti_pose/iter_82.npy

# ### eyeglasses evaluation ### ## motion oven --> arti_pose/motion_oven ###
# # export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_oven"
# export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_eyeglasses"
# export pre_compute_delta=0
# export pre_compute_delta=2
# export cuda_ids=0

# # CUDA_VISIBLE_DEVICES=${cuda_ids}  TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=480 --bsz=1 --nmasks=3 --use-equi=35 --part-pred-npoints=100 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="eyeglasses" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv18" --kpconv-kanchor=60  --sel-mode=29 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=1 --resume-path-glb=/share/xueyi/ckpt/playground/model_20220325_15:36:20/ckpt/playground_net_Iter9800.pth  --pred-axis=1  --mtx-based-axis-regression=True --with-part-proposal=False  --run-mode=eval --pre-compute-delta=1 --sel-mode-trans=3 --resume-path=/share/xueyi/ckpt/playground/model_20220515_10:45:41/ckpt/playground_net_Iter800.pth

# #### ==== eyeglasses eval ==== ####

# CUDA_VISIBLE_DEVICES=${cuda_ids}  TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=512 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=256 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='hoi4d' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="safe" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv19" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=/share/xueyi/ckpt/playground/model_20220513_01:55:56/ckpt/playground_net_Iter5000.pth   --run-mode=eval --pre-compute-delta=1 --sel-mode-trans=52 --resume-path=/share/xueyi/ckpt/playground/model_20220516_09:35:01/ckpt/playground_net_Iter200.pth  --mtx-based-axis-regression=False
# ### eyeglasses evaluation ### ## motion oven --> arti_pose/motion_oven ###
# #### ==== eyeglasses eval ==== ####