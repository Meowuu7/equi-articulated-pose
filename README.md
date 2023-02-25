# Equi-Articulated-Pose

This is the official code repo of the paper:

[Self-Supervised Category-Level Articulated Object Pose Estimation with Part-Level SE(3) Equivariance](https://equi-articulated-pose.github.io/), *Xueyi Liu*, *Ji Zhang*, *Ruizhen Hu*, *Haibin Huang*, *He Wang*, *Li Yi*, ICLR 2023.

![overall_pipeline](./assets/overall-pipeline-z.pdf)

## Links

- [Project Page](https://equi-articulated-pose.github.io/)
- [Openreview Page](https://openreview.net/forum?id=20GtJ6hIaPA)

## TODOs

- [x] Training and test code
- [x] Environments
- [ ] Data
- [ ] Add the arXiv Page

## Environment and package dependency

Create a virtual environment: 
```shell
conda env create -f env.yaml
```

Install the vgtk package:
```shell
cd vgtk && python setup.py install && cd ..
```

## Data

TODO

## Training

```shell
bash scripts/train/${CATEGORY_NAME}.sh
```

## Evaluation

```shell
bash scripts/val/${CATEGORY_NAME}.sh
```


## License

(Add licenses here)

Our code and data are released under MIT License (see LICENSE file for details).


## Reference

(Add references here like follows)

EPN,equipose

Part of the code is taken from [EPN](https://github.com/SimingYan/HPNet), [equi-pose](https://github.com/lingxiaoli94/SPFN).

[1] xxx

[2] xxx
