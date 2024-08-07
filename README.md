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

### Under GED metric
To train and test on AIDS700nef.
```
python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16
```
To train and test on IMDBMulti.
```
python src/main.py --gnn rggc --k 4 --epochs 30000 --dataset IMDBMulti --use-pe --pe-dim 16
```
To train and test on LINUX.
```
python src/main.py --k 8 --learning 0.002 --epochs 20000 --dataset LINUX --use-pe --pe-dim 10
```
To train and test on PTC.
```
python src/main.py --gnn rggc --k 8 --epochs 5000 --dataset PTC --use-pe --pe-dim 20
```

### Under MCS metric
To train and test on AIDS700nef.
```
python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16 --metric mcs
```
To train and test on IMDBMulti.
```
python src/main.py --gnn rggc --k 4 --epochs 30000 --dataset IMDBMulti --use-pe --pe-dim 16 --metric mcs
```
To train and test on LINUX.
```
python src/main.py --gnn rggc --k 8 --learning 0.002 --epochs 20000 --dataset LINUX --use-pe --pe-dim 10 --metric mcs
```
To train and test on PTC.
```
python src/main.py --gnn rggc --k 8 --epochs 20000 --dataset PTC --use-pe --pe-dim 20 --metric mcs
```