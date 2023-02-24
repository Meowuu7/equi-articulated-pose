# from SPConvNets.trainer_modelnet import Trainer

# import options
from SPConvNets.options import opt

import torch

if __name__ == '__main__':
    opt.model.flag = 'regular'
    # opt.model.model = "cls_so3net_pn"
    # opt.model.model = "cls_so3posenet_pn"
    # opt.model.model = "unsup_seg_so3posenet_pn"
    # opt.model.model = "unsup_seg_so3posenet_pn_2"
    # opt.model.model = "unsup_seg_basic_conv_pn_2"
    if opt.equi_settings.use_equi == 2:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_2"
    elif opt.equi_settings.use_equi == 3:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_3"
    elif opt.equi_settings.use_equi == 4:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_4"
    elif opt.equi_settings.use_equi == 5:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_5"
    elif opt.equi_settings.use_equi == 6:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_6"
    elif opt.equi_settings.use_equi == 9:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_9"
    elif opt.equi_settings.use_equi == 10:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_10"
    elif opt.equi_settings.use_equi == 11:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_11"
    elif opt.equi_settings.use_equi == 12:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_12"
    elif opt.equi_settings.use_equi == 13:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_13"
    elif opt.equi_settings.use_equi == 14:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_14"
    elif opt.equi_settings.use_equi == 16:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_16"
    elif opt.equi_settings.use_equi == 17:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_17"
    elif opt.equi_settings.use_equi == 18:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_18"
    elif opt.equi_settings.use_equi == 19:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_19"
    elif opt.equi_settings.use_equi == 20:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_20"
    elif opt.equi_settings.use_equi == 21:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_21"
    elif opt.equi_settings.use_equi == 22:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_22"
    elif opt.equi_settings.use_equi == 23:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_23"
    elif opt.equi_settings.use_equi == 24:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_24"
    elif opt.equi_settings.use_equi == 25:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_25"
    elif opt.equi_settings.use_equi == 26:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_26"
    elif opt.equi_settings.use_equi == 27:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_27"
    elif opt.equi_settings.use_equi == 28:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_28"
    elif opt.equi_settings.use_equi == 29:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_29"
    elif opt.equi_settings.use_equi == 30:
        opt.model.model = "unsup_seg_so3_pose_conv_pn_30_single_stage"
    elif opt.equi_settings.use_equi == 1:
        opt.model.model = "unsup_seg_so3_conv_pn_2"
    else:
        opt.model.model = "unsup_seg_basic_conv_pn_2"

    if torch.cuda.device_count() > 1:
        print("Using trainer in multi-gpu version!")
        from SPConvNets.trainer_unsup_seg_mg import Trainer
    else:
        print("Using Trainer in single-gpu version!")
        # from SPConvNets.trainer_unsup_ToySeg import Trainer
        from SPConvNets.trainer_unsup_seg_mg import Trainer

    if opt.mode == 'train':
        # overriding training parameters here
        # opt.batch_size = 32 # 4 #6 #12
        opt.batch_size = 2 # 4 #6 #12
        opt.batch_size = opt.equi_settings.bsz
        opt.train_lr.decay_rate = 0.5
        opt.train_lr.decay_step = 20000
        opt.train_loss.attention_loss_type = 'default'
    opt.batch_size = opt.equi_settings.bsz
    print("here1")
    trainer = Trainer(opt)
    print("here2 tariner constructed")
    trainer.mode = "train"
    trainer.mode =  opt.mode # "train"
    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        print("here eval")
        trainer.eval()
