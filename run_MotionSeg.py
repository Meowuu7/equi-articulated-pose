# from SPConvNets.trainer_modelnet import Trainer
from SPConvNets.trainer_MotionSeg import Trainer
from SPConvNets.options import opt

if __name__ == '__main__':
    opt.model.flag = 'regular'
    # opt.model.model = "cls_so3net_pn"
    opt.model.model = "cls_so3posenet_pn"

    if opt.mode == 'train':
        # overriding training parameters here
        opt.batch_size = 4 # 4 #6 #12
        opt.train_lr.decay_rate = 0.5
        opt.train_lr.decay_step = 20000
        opt.train_loss.attention_loss_type = 'default'

    trainer = Trainer(opt)
    if opt.mode == 'train':
        trainer.train()
    elif opt.mode == 'eval':
        trainer.eval()
