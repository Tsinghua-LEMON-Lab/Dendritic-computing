#Power-efficient neural network with artificial dendrites

## Environmental requirements
```
python 3.8.5
pytorch 1.8.0
torchvision 0.9.0
```

## The steps are as follows
### Training：
```bash
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100 --pretrained_path results/ann/svhn/checkpoint/last.pth
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100 --pretrained_path results/ann/svhn/checkpoint/last.pth
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100 --pretrained_path results/ann/svhn/checkpoint/last.pth
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.0005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100 --pretrained_path results/ann/svhn/checkpoint/last.pth
CUDA_VISIBLE_DEVICES=2 python main.py --num_workers 4  --lr 0.0005   --batch_size 128  --scheduler multi_step   --save_root results   --num_epochs  100 --pretrained_path results/ann/svhn/checkpoint/last.pth 
```

### Test：
```bash
CUDA_VISIBLE_DEVICES=2  python main.py --num_workers 4  --batch_size 128  --eval True --pretrained_path results/ann/svhn/checkpoint/last.pth
```