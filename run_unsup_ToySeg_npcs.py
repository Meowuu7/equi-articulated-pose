# from SPConvNets.trainer_modelnet import Trainer

from SPConvNets.options import opt

import torch

if __name__ == '__main__':
    opt.model.flag = 'regular'

    opt.model.model = "unsup_seg_so3_pose_conv_pn_8"

    from SPConvNets.trainer_unsup_seg_mg_npcs import Trainer

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
