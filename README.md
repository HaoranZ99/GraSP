# GraSP
## Requirements
```
matplotlib        3.7.1
networkx          2.8.4
numpy             1.23.5
pandas            1.5.3
pyg               2.1.0
pytorch           1.11.0
pytorch-cluster   1.6.0
pytorch-scatter   2.0.9
pytorch-sparse    0.6.15
scikit-learn      1.2.2
scipy             1.10.1
tqdm              4.65.0
```
## Examples on Training and Inference

```
python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16
python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16 --metric mcs
```

## Reproductivity
To reproduce our empirical results, run the ``run.sh`` script.
```
bash run.sh
```

## Acknowledgement
We thank [SimGNN](https://github.com/benedekrozemberczki/SimGNN) and [Extended-SimGNN](https://github.com/gospodima/Extended-SimGNN) for providing the core frameworks utilized in this implementation.