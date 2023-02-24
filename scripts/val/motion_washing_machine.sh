### oven evaluation ### ## motion oven --> arti_pose/motion_oven ###
# export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_oven"
export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_washing_machine"
export pre_compute_delta=0
export pre_compute_delta=2
export cuda_ids=2,3,5,6,7
export num_gpus=5
export resume_path_glb="/share/xueyi/ckpt/playground/model_20220327_07:06:40/ckpt/playground_net_Iter3400.pth"
export resume_path="/share/xueyi/ckpt/playground/model_20230224_17:25:46/ckpt/playground_net_Iter600.pth"

export input_num=512
export input_num=380

export part_pred_npoints=256
export part_pred_npoints=128


export input_num=512
export input_num=380
# 
export part_pred_npoints=256
export part_pred_npoints=128

export cuda_ids=2,3,4,5,6,7
export num_gpus=6

# 

## motion oven ##
### nn_iters should be 2 ### # should be 2 ##
CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=${num_gpus} run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=2 --global-rot=1 --input-num=${input_num} --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=${part_pred_npoints} --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="washing_machine" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=6 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="pvp_equi_rot_whl_shp_inv19" --kpconv-kanchor=60  --sel-mode=-1 --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=${resume_path_glb} --mtx-based-axis-regression=False  --sel-mode-trans=-1 \
--run-mode=eval --pre-compute-delta=1 --sel-mode-trans=-1   --resume-path=${resume_path}

# one iter; washing machine #
# ckpts: /share/xueyi/ckpt/playground/model_20230224_16:19:25/ckpt/playground_net_Iter200.pth

# two iter; washing machine #
# ckpts: /share/xueyi/ckpt/playground/model_20230224_17:25:46/ckpt/playground_net_Iter200.pth

## for eval ##
#  --run-mode=eval --pre-compute-delta=1 --sel-mode-trans=-1   --resume-path=/share/xueyi/ckpt/playground/model_20220515_12:17:16/ckpt/playground_net_Iter400.pth  --mtx-based-axis-regression=False
