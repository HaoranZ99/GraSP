python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16
python src/main.py --gnn rggc --k 4 --epochs 30000 --dataset IMDBMulti --use-pe --pe-dim 16
python src/main.py --gnn rggc --k 8 --learning 0.002 --epochs 20000 --dataset LINUX --use-pe --pe-dim 10
python src/main.py --gnn rggc --k 8 --epochs 5000 --dataset PTC --use-pe --pe-dim 20

python src/main.py --gnn rggc --k 8 --epochs 30000 --use-pe --pe-dim 16 --metric mcs
python src/main.py --gnn rggc --k 4 --epochs 30000 --dataset IMDBMulti --use-pe --pe-dim 16 --metric mcs
python src/main.py --gnn rggc --k 8 --learning 0.0015 --epochs 30000 --dataset LINUX --use-pe --pe-dim 10 --metric mcs
rm -r datasets/PTC/*  # Remove the previous split and download the split corresponding to the MCS metric
python src/main.py --gnn rggc --k 8 --epochs 20000 --dataset PTC --use-pe --pe-dim 20 --metric mcs