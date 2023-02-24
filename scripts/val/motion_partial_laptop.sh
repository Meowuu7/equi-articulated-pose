## laptop evaluation ###
# export eval_data_sv_dict_fn="/data1/sim/equi_arti_pose/motion_partial_laptop"
export eval_data_sv_dict_fn="./sv_motion_partial_laptop"
export resume_path_glb="/share/xueyi/ckpt/playground/model_20220417_03:43:45/ckpt/playground_net_Iter800.pth"
export resume_path="/share/xueyi/ckpt/playground/model_20220508_18:22:54/ckpt/playground_net_Iter600.pth"
export sel_mode=29
export pre_compute_delta=1
export cuda_ids=7


CUDA_VISIBLE_DEVICES=${cuda_ids} TORCH_DISTRIBUTED_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=1 run_unsup_ToySeg_multi_stage.py experiment -d "./data" --init-lr=1e-4 --num-iters=1 --global-rot=1 --input-num=380 --bsz=1 --nmasks=2 --use-equi=38 --part-pred-npoints=128 --inv-attn=1 --orbit-attn=0 --slot-iters=7 --dataset-type='motion_partial' --rot-factor=0.5 --init-radius=0.20 --equi-anchors=1  --translation=0 --rot-anchors=0 --gt-oracle-seg=0 --no-articulation=0 --gt-oracle-trans=0 --kanchor=60 --cent-trans=3 --shape-type="laptop" --soft-attn=1 --feat-pooling=mean --factor=0.99 --queue-len=200 --glb-recon-factor=1.0 --slot-recon-factor=0.5 --use-sigmoid=1 --recon-prior=9 --lr-adjust=2 --n-dec-steps=1000 --use-flow-reg=0 --save-freq=200 --use-multi-sample=1 --n-samples=100 --exp-indicator="motion_partial_laptop_eval" --kpconv-kanchor=60  --sel-mode=${sel_mode} --slot-single-mode=1 --cur-stage=1 --permute-modes=1 --use-2d=0 --pred-axis=1 --resume-path-glb=${resume_path_glb}   --run-mode=eval --pre-compute-delta=${pre_compute_delta} --sel-mode-trans=14 --resume-path=${resume_path} --mtx-based-axis-regression=False --eval_data_sv_dict_fn=${eval_data_sv_dict_fn}
