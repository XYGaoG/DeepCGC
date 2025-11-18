# DeepCGC: Unveiling the Deep Clustering Mechanism of Fast Graph Condensation

## Introduction
This repository contains the official implementation of our paper **DeepCGC: Unveiling the Deep Clustering Mechanism of Fast Graph Condensation**.

DeepCGC extends CGC ([code](https://arxiv.org/abs/2405.13707)) from our WWW 2025 paper [Rethinking and Accelerating Graph Condensation: A Training-Free Approach with Class Partition](https://arxiv.org/abs/2405.13707). 

We generalize CGC's class-to-node matching principle into a broader latent-space formulation, revealing that graph condensation can be interpreted as a class-wise clustering problem in the latent space.

<p align="center">
<img src="figure.png" alt="GC" width="500">
</p>

The key improvements of DeepCGC include:
- 🎯 **Clustering-driven optimization objective**
- 🔄 **Non-linear, invertible relay model**  
- 💪 **Enhanced representational capacity while maintaining efficiency**


For more works about graph condensation, please refer to our TKDE'25 survey paper 🔥[Graph Condensation: A Survey](https://arxiv.org/abs/2401.11720v2) and paper list [Graph Condensation Papers](https://github.com/XYGaoG/Graph-Condensation-Papers).



### Requirements
Required dependencies are provided in `./requirements.txt`.


## Dataset

Configure the dataset directory path via `args.raw_data_dir` and ensure all datasets are downloaded to this location.

* For Cora and Citeseer, they will be downloaded from [PYG](https://www.pyg.org/).
* Ogbn-products will be downloaded from [OGB](https://ogb.stanford.edu/docs/nodeprop/).
* For Ogbn-arxiv, Flickr and Reddit, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). Note that the links are provided by GraphSAINT team. 


## Condensation

To condense the graph using DeepCGC and train GCN models:

```bash
$ python main.py --gpu 0 --dataset reddit --ratio 0.001 --generate_adj 1
```

For more efficient graphless variant DeepCGC-X: 
```bash
$ python main.py --gpu 0 --dataset reddit --ratio 0.001 --generate_adj 0
```

Results will be saved in `./results/`. The condensed graph will be saved in `./cond_graph/`.

### Additional Scripts

Comprehensive scripts for different condensation ratios are provided in `./script.sh`. 



## Hyper-parameters

For pre-defined condensation ratios in `./script.sh`, hyperparameters are automatically loaded from `./config/dataset_name.yaml`.

For custom condensation ratios, we recommend hyperparameter search on the validation set for optimal performance.


## Citation

```
@inproceedings{gao2025rethinking,
 title={Rethinking and Accelerating Graph Condensation: A Training-Free Approach with Class Partition},
 author={Gao, Xinyi and Ye, Guanhua and Chen, Tong and Zhang, Wentao and Yu, Junliang and Yin, Hongzhi},
 booktitle={Proceedings of the ACM on Web Conference 2025},
 year={2025}
}
```

```
@article{gao2025graph,
  title={Graph condensation: A survey},
  author={Gao, Xinyi and Yu, Junliang and Chen, Tong and Ye, Guanhua and Zhang, Wentao and Yin, Hongzhi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  publisher={IEEE}
}

```
