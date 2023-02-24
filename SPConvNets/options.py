
import vgtk


parser = vgtk.HierarchyArgmentParser()

# Experiment arguments
exp_args = parser.add_parser("experiment")
exp_args.add_argument('--experiment-id', type=str, default='playground',
                      help='experiment id')
exp_args.add_argument('-d', '--dataset-path', type=str, required=True,
                      help='path to datasets')
exp_args.add_argument('--dataset', type=str, default='kpts',
                      help='name of the datasets')
exp_args.add_argument('--model-dir', type=str, default='/share/xueyi/ckpt',
                      help='path to models')
exp_args.add_argument('-s', '--seed', type=int, default=2913,
                      help='random seed')
exp_args.add_argument('--run-mode', type=str, default='train',
                      help='train | eval | test')

# Network arguments
net_args = parser.add_parser("model")
net_args.add_argument('-m', '--model', type=str, default='inv_so3net_pn',
                      help='type of model to use')
net_args.add_argument('--input-num', type=int, default=1024,
                      help='the number of the input points')
net_args.add_argument('--output-num', type=int, default=32,
                      help='the number of the input points')
net_args.add_argument('--search-radius', type=float, default=0.4)
net_args.add_argument('--normalize-input', action='store_true',
                      help='normalize the input points')
net_args.add_argument('--dropout-rate', type=float, default=0.,
                      help='dropout rate, no dropout if set to 0')
net_args.add_argument('--init-method', type=str, default="xavier",
                      help='method for weight initialization')
net_args.add_argument('-k','--kpconv', action='store_true', default=False, help='If set, use a kpconv structure instead')
net_args.add_argument('--kanchor', type=int, default=20, help='# of anchors used: {1,20,40,60}') # rotation anchor poit
net_args.add_argument('--normals', action='store_true', help='If set, add normals to the input (default setting is false)')
net_args.add_argument('-u', '--flag', type=str, default='max',
                      help='pooling method: max | mean | attention | rotation')
net_args.add_argument('--representation', type=str, default='quat',
                      help='how to represent rotation: quaternion | ortho6d ')



# Training arguments
train_args = parser.add_parser("train")
train_args.add_argument('-e', '--num-epochs', type=int, default=None,
                        help='maximum number of training epochs')
train_args.add_argument('-i', '--num-iterations', type=int, default=1000000,
                        help='maximum number of training iterations')
train_args.add_argument('-b', '--batch-size', type=int, default=8,
                        help='batch size to train')
train_args.add_argument('--npt', type=int, default=24,
                        help='number of point per fragment')
train_args.add_argument('-t', '--num-thread', default=8, type=int,
                        help='number of threads for loading data')
train_args.add_argument('--no-augmentation', action="store_true",
                        help='no data augmentation if set true')
train_args.add_argument('-r','--resume-path', type=str, default=None,
                        help='Training using the pre-trained model')
train_args.add_argument('-rglb','--resume-path-glb', type=str, default=None,
                        help='Training using the pre-trained model')
train_args.add_argument('--save-freq', type=int, default=5000,
                        help='the frequency of saving the checkpoint (iters)')
train_args.add_argument('-lf','--log-freq', type=int, default=100,
                        help='the frequency of logging training info (iters)')
train_args.add_argument('--eval-freq', type=int, default=5000,
                        help='frequency of evaluation (iters)')
train_args.add_argument('--debug-mode', type=str, default=None,
                        help='if specified, train with a certain debug procedure')


# Learning rate arguments
lr_args = parser.add_parser("train_lr")
lr_args.add_argument('-lr', '--init-lr', type=float, default=1e-4,
                     help='the initial learning rate')
lr_args.add_argument('-lrt', '--lr-type', type=str, default='exp_decay',
                     help='learning rate schedule type: exp_decay | constant')
lr_args.add_argument('--decay-rate', type=float, default=0.5,
                     help='the rate of exponential learning rate decaying')
lr_args.add_argument('--decay-step', type=int, default=10000,
                     help='the frequency of exponential learning rate decaying')
# lr_args.add_argument('-nmasks', '--nmasks', type=int, default=5,
#                      help='the initial learning rate')

# Loss funtion arguments
loss_args = parser.add_parser("train_loss")
loss_args.add_argument('--loss-type', type=str, default='soft',
                       help='type of loss function')
loss_args.add_argument('--attention-loss-type', type=str, default='no_reg',
                       help='type of attention loss function')
loss_args.add_argument('--margin', type=float, default=1.0,
                       help='margin of hard batch loss')
loss_args.add_argument('--temperature', type=float, default=3,
                       help='margin of hard batch loss')
loss_args.add_argument('--attention-margin', type=float, default=1.0,
                       help='margin of attention loss')
loss_args.add_argument('--attention-pretrain-step', type=int, default=3000,
                       help='step for scheduled pretrain (only used in attention model)')
loss_args.add_argument('--equi-alpha', type=float, default=0.0,
                       help='weight for equivariance loss')
# loss_args.add_argument('---alpha', type=float, default=0.0,
#                        help='weight for equivariance loss')


loss_args = parser.add_parser("equi_settings")
loss_args.add_argument('--num-iters', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--global-rot', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--nmasks', type=int, default=4,
                       help='type of loss function')
loss_args.add_argument('--use-equi', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--bsz', type=int, default=32,
                       help='type of loss function')
loss_args.add_argument('--part-pred-npoints', type=int, default=128,
                       help='type of loss function')
loss_args.add_argument('--model-type', type=str, default='so3pose',
                       help='type of loss function')
loss_args.add_argument('--decoder-type', type=str, default='regular',
                       help='type of loss function')
loss_args.add_argument('--inv-attn', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--orbit-attn', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--topk', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--slot-iters', type=int, default=3,
                       help='type of loss function')
loss_args.add_argument('--dataset-type', type=str, default='partnet',
                       help='type of loss function')
loss_args.add_argument('--rot-factor', type=float, default=1.0,
                       help='type of loss function')
loss_args.add_argument('--init-radius', type=float, default=0.2,
                       help='type of loss function')
loss_args.add_argument('--rot-anchors', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--translation', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--gt-oracle-seg', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--no-articulation', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--gt-oracle-trans', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--feat-pooling', type=str, default='mean',
                       help='type of loss function')
loss_args.add_argument('--cent-trans', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--shape-type', type=str, default="eyeglasses",
                       help='type of loss function')
loss_args.add_argument('--soft-attn', type=int, default=0,
                       help='type of loss function')
# loss_args.add_argument('--', type=int, default=0,
#                        help='type of loss function')
loss_args.add_argument('--recon-prior', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--factor', type=float, default=0.9,
                       help='type of loss function')
loss_args.add_argument('--queue-len', type=int, default=200,
                       help='type of loss function')
# loss_args.add_argument('--factor', type=float, default=0.9,
#                        help='type of loss function')

loss_args.add_argument('--glb-recon-factor', type=float, default=2.0,
                       help='type of loss function')
loss_args.add_argument('--slot-recon-factor', type=float, default=4.0,
                       help='type of loss function')
loss_args.add_argument('--use-sigmoid', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--lr-adjust', type=int, default=2,
                       help='type of loss function')
loss_args.add_argument('--n-dec-steps', type=int, default=20,
                       help='type of loss function')
loss_args.add_argument('--lr-decay-factor', type=float, default=0.7,
                       help='type of loss function')
loss_args.add_argument('--use-flow-reg', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--pre-compute-delta', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--use-multi-sample', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--n-samples', type=int, default=100,
                       help='type of loss function')
loss_args.add_argument('--partial', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--use-axis-queue', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--exp-indicator', type=str, default='xxx',
                       help='type of loss function')
loss_args.add_argument('--loss-weight-reg', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--est-normals', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--kpconv-kanchor', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--cur-stage', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--r-representation', type=str, default='quat',
                       help='type of loss function')
loss_args.add_argument('--slot-single-mode', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--sel-mode', type=int, default=-1,
                       help='type of loss function')
loss_args.add_argument('--sel-mode-trans', type=int, default=-1,
                       help='type of loss function')
loss_args.add_argument('--permute-modes', type=int, default=1,
                       help='type of loss function')
loss_args.add_argument('--use-2d', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--use-inv-angles', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--rot-angle-factor', type=float, default=0.5,
                       help='type of loss function')
loss_args.add_argument('--pred-axis', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--pred-pv-equiv', type=int, default=0,
                       help='type of loss function')
loss_args.add_argument('--mtx-based-axis-regression', type=bool, default=False,
                       help='type of loss function')
loss_args.add_argument('--axis-reg-stra', type=int, default=0,
                       help='type of loss function')  # axis_reg_stra # --axis-reg-stra=1
loss_args.add_argument('--glb-single-cd', type=int, default=0,
                       help='type of loss function') # slot_single_cd
loss_args.add_argument('--slot-single-cd', type=int, default=0,
                       help='type of loss function') # slot_single_cd
loss_args.add_argument('--rel-for-points', type=int, default=0,
                       help='type of loss function') # slot_single_cd
loss_args.add_argument('--feat-partition', type=int, default=0,
                       help='type of loss function') # slot_single_cd
loss_args.add_argument('--use-art-mode', type=bool, default=False,
                       help='type of loss function')
loss_args.add_argument('--with-part-proposal', type=bool, default=True, # with_part_proposal
                       help='type of loss function')
loss_args.add_argument('--add-normal-noise', type=float, default=-1,
                       help='type of loss function') # slot_single_cd # add_normal_noise
# eval_data_sv_dict # eval_data_sv_dict_fn
loss_args.add_argument('--eval_data_sv_dict_fn', type=str, default="/data1/sim/equi_arti_pose",
                       help='type of loss function') # slot_single_cd # add_normal_noise

loss_args = parser.add_parser("parallel")
loss_args.add_argument('--local_rank', type=int, default=0,
                       help='type of loss function')



# Eval arguments
eval_args = parser.add_parser("eval")

# Test arguments
test_args = parser.add_parser("test")

# Tog seg sampling options


opt = parser.parse_args()


opt.mode = opt.run_mode

# self.z_len_min = opt.z_len_min
#         self.z_len_max = opt.z_len_max
#         self.z_len2_min = opt.z_len2_min
#         self.z_len2_max = opt.z_len2_max
#         self.x_len_min = opt.x_len_min
#         self.x_len_max = opt.x_len_max
#         self.y_len_min = opt.y_len_min
#         self.y_len_max = opt.y_len_max
#         self.y_len2_min = opt.y_len2_min
#         self.y_len2_max = opt.y_len2_max
#         self.num_points = opt.num_points
#         self.up_p_ratio_min = opt.up_p_ratio_min
#         self.up_p_ratio_max = opt.up_p_ratio_max

# add sampling options for ToySeg model
opt.z_len_min = 50
opt.z_len_max = 100
opt.z_len2_min = 30
opt.z_len2_max = 100
opt.x_len_min = 40
opt.x_len_max = 200
opt.y_len_min = 3
opt.y_len_max = 10
opt.y_len2_min = 5
opt.y_len2_max = 10
opt.num_points = 128 # 256 #  512 # 1024
opt.num_points =  512 # 1024
opt.num_points =  256 # 1024
# opt.num_points = 1024
# opt.model.input_num = opt.num_points
opt.num_points = opt.model.input_num
opt.up_p_ratio_min = 0.3
opt.up_p_ratio_max = 0.7

opt.train_len = 50000
opt.test_len = 10000
# opt.nmasks = 5

# 128 x 4 ---- enough for us to reconstruct the whole shape; how to do part-by-part reconstruction?
# need to cluster points to different parts and use per-part representations to reconstruct the shape
# opt.nmasks = 4
opt.nmasks = 10
opt.nmasks = opt.equi_settings.nmasks
# opt.nmasks = 2
# opt.nmasks = 4
# opt.nmasks = 3


# todo: should adjust some parameters: dropout_rate