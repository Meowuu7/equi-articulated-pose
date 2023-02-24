# Equi-Articulated-Pose

This is the official code repo of the paper:

[Self-Supervised Category-Level Articulated Object Pose Estimation with Part-Level SE(3) Equivariance](https://equi-articulated-pose.github.io/), *Xueyi Liu*, *Ji Zhang*, *Ruizhen Hu*, *Haibin Huang*, *He Wang*, *Li Yi*, ICLR 2023.

![Screen Shot 2022-04-14 at 5.46.04 PM](./assets/overall-pipeline-23-1.png)

## Links

- [Project Page](https://equi-articulated-pose.github.io/)
- [arXiv Page](https://arxiv.org/abs/2203.06558)

## Environment and package dependency

The main experiments are implemented on PyTorch 1.9.1, Python 3.8.8. Main dependency packages are listed as follows:

```
torch_cluster==1.5.9
torch_scatter==2.0.7
horovod==0.23.0
pykdtree==1.3.4
numpy==1.20.1
h5py==2.8.0
```

## Mobility-based part segmentation

### Data

```shell
cd data
mkdir part-segmentation
```

Download data from [here](https://drive.google.com/file/d/1XTjkFqOs-wbnQ90aFqxsxsH8ii80mmlR/view?usp=sharing) and put the file under the `data/part-segmentation` folder. 

Unzip the downloaded data and zipped files under its subfolders as well. 

### Supervision search stage

To create and optimize the intermediate supervision space for the mobility-based part segmentation task, please use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/motion_seg_h_mb_cross.yaml
```


The default backbone is DGCNN. 

### Sample supervision features from the optimized supervision feature distribution space

The following command samples a set of operations with relatively high sampling probabilities from the optimized supervision space (distribution parameters are stored in `dist_params.npy` under the logging directory):

```shell
python load_and_sample.py -c cfgs/${your_config_file} --ty=loss --params-path=${your_parameter_file}
```

### Training stage

Insert supervisions to use in the corresponding trainer file and use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/motion_seg_h_mb_cross_tst_perf.yaml
```

### Test stage

Change the `resume` argument in the `./cfgs/motion_seg_h_mb_cross_pure_test_perf.yaml` file to the saved model's checkpoint to test and run the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np 1  -H ${your_machine_ip}:1 python -W ignore main_prm.py -c ./cfgs/motion_seg_h_mb_cross_pure_test_perf.yaml
```

### Checkpoints

Please download optimized distribution parameters and trained models from [here](https://drive.google.com/drive/folders/1oPocnUABlkRbO9wmwmKHCy2VM-BZrUDm?usp=sharing).

### Comments

- We change the number of segmentations sampled for each training shape from at most 5 to at most 2 for release.
- We test on 4 GPUs.

## Primitive fitting

### Data

- Step 1: Download the Traceparts data from [SPFN](https://github.com/lingxiaoli94/SPFN) repo. Put it under the `data/` folder. 
- Step 2: Download data splitting from [Data Split](https://drive.google.com/file/d/1bp2NGcV4ST6Fb2flBAHZsFvj_wnFpZ9j/view?usp=sharing). Put it under the `data/traceparts_data` folder. Unzip the zip file to get data splitting files. 

### Supervision search stage

To create and optimize the intermediate supervision space for the primitive fitting task, please use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_mb_cross_v2_tree.yaml
```

The default backbone is DGCNN.

To optimize the supervision distribution space for the first stage of primitive fitting task using HPNet-style network architecture, please use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_mb_v2_tree_optim_loss.yaml
```

### Sample supervision features from the optimized supervision feature distribution space

The following command samples a set of operations with relatively high sampling probabilities from the optimized supervision space (distribution parameters are stored in `dist_params.npy` under the logging directory):

```shell
python load_and_sample.py -c cfgs/${your_config_file} --ty=loss --params-path=${your_parameter_file}
```

### Training stage

Insert supervisions to use in the corresponding trainer file and use the following command:

```shell
CUDA_VISIBLE_DEVICES=${devices} horovodrun -np ${n_device}  -H ${your_machine_ip}:${n_device} python -W ignore main_prm.py -c ./cfgs/prim_seg_h_ob_v2_tree.yaml
```

### Inference stage

Replace `resume` in `prim_inference.yaml` to the path to saved model weights and use the following command to evaluate the trained model:

```shell
python -W ignore main_prm.py -c ./cfgs/prim_inference.yaml
```

Remember to select a free GPU in the config file.

You should modify the file `prim_inference.py` to choose whether to use the clustering-based segmentation module or classification-based one.

For clustering-based segmentation, use the `_clustering_test` function; For another, use the `_test` function.

### Checkpoints

Please download optimized distribution parameters from [here](https://drive.google.com/drive/folders/1bDF81h-ATSdiejnU888f7IihEkNhXH9r?usp=sharing).

## Unsolved Problems

- ***Unfriendly operations***: Sometimes the model will sample operations which would result in a supervision feature with very large absolute values. It would scarcely hinder the optimization process (since such supervisions would cause low metric values; thus, the model using them will not be passed to the next step), making the optimization process ugly. 

  The problem could probably be solved by forbidding certain operation combinations/sequences. And please feel free to submit a pull request if you can solve it. 

- ***Unnormalized rewards***: Reward values used for such three tasks may have different scales. It may affect the optimization process to some extent. They could probably be normalized using prior knowledge of generalization gaps of each task and its corresponding training data. 




## License

Our code and data are released under MIT License (see LICENSE file for details).


## Reference

Part of the code is taken from [HPNet](https://github.com/SimingYan/HPNet), [SPFN](https://github.com/lingxiaoli94/SPFN), [Deep Part Induction](https://github.com/ericyi/articulated-part-induction), [PointNet2](https://github.com/charlesq34/pointnet2), [MixStyle](https://github.com/KaiyangZhou/mixstyle-release).

[1] Yi, L., Kim, V. G., Ceylan, D., Shen, I. C., Yan, M., Su, H., ... & Guibas, L. (2016). A scalable active framework for region annotation in 3d shape collections. *ACM Transactions on Graphics (ToG)*, *35*(6), 1-12.

[2] Mo, K., Zhu, S., Chang, A. X., Yi, L., Tripathi, S., Guibas, L. J., & Su, H. (2019). Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 909-918).

[3] Yi, L., Huang, H., Liu, D., Kalogerakis, E., Su, H., & Guibas, L. (2018). Deep part induction from articulated object pairs. *arXiv preprint arXiv:1809.07417*.

[4] Li, L., Sung, M., Dubrovina, A., Yi, L., & Guibas, L. J. (2019). Supervised fitting of geometric primitives to 3d point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2652-2660).