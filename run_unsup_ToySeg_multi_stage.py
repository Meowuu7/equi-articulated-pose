# from SPConvNets.trainer_modelnet import Trainer

# import options
from SPConvNets.options import opt

import torch

if __name__ == '__main__':
    opt.model.flag = 'regular'

    if opt.equi_settings.use_equi == 27:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_27_multi_stage"
    elif opt.equi_settings.use_equi == 31:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_31_multi_stage"
    elif opt.equi_settings.use_equi == 32:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_32_multi_stage"
    elif opt.equi_settings.use_equi == 33:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_33_multi_stage"
    elif opt.equi_settings.use_equi == 34:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_34_multi_stage"
    elif opt.equi_settings.use_equi == 35:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_35_multi_stage"
    elif opt.equi_settings.use_equi == 36:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_36_multi_stage"
    elif opt.equi_settings.use_equi == 38:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_38_multi_stage"
    elif opt.equi_settings.use_equi == 39:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_39_multi_stage"
    elif opt.equi_settings.use_equi == 40:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_40_multi_stage"
    else:
        raise ValueError(f"Unrecognized use_equi: {opt.equi_settings.use_equi}!!!")

    from SPConvNets.trainer_unsup_seg_mg_multi_stage import Trainer

    if opt.mode == 'train':
        # overriding training parameters here
        # opt.batch_size = 32 # 4 #6 #12
        opt.batch_size = 2 # 4 #6 #12
        opt.batch_size = opt.equi_settings.bsz
        opt.train_lr.decay_rate = 0.5
        opt.train_lr.decay_step = 20000
        opt.train_loss.attention_loss_type = 'default'
    opt.batch_size = opt.equi_settings.bsz
    trainer = Trainer(opt)
    trainer.mode = "train"
    trainer.mode =  opt.mode # "train"
    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        print("here eval")
        trainer.eval()