export CUDA_VISIBLE_DEVICES=2
python -W ignore rl_quantize.py     \
 --arch resnet50                    \
 --dataset cifar10                 \
 --suffix ratio010                  \
 --preserve_ratio 0.27            \
 --n_worker 32                      \
 --data_bsize 256                   \
 --train_size 20000                 \
 --val_size 10000                   \
 --output "save" \
 --min_bit 8 \
 --max_bit 32 \
 --dataset_root ../Datasets/cifar10
#  --preserve_ratio 0.27

