python train.py --dataset_name LA --labelnum 16 --gpu 0 && \
python train.py --dataset_name LA --labelnum 8 --gpu 0 && \
python train.py --dataset_name LA --labelnum 4 --gpu 0 && \

python train.py --dataset_name Pancreas_CT --labelnum 12 --gpu 0 && \
python train.py --dataset_name Pancreas_CT --labelnum 6 --gpu 0 && \
python train.py --dataset_name Pancreas_CT --labelnum 3 --gpu 0 && \

python train.py --dataset_name BraTS2019 --labelnum 25 --gpu 0 && \
python train.py --dataset_name BraTS2019 --labelnum 12 --gpu 0
