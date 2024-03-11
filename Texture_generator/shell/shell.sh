# BraTS Train
python main.py --config configs/brats-LBBDM-f2.yaml --train --sample_at_start --save_top --gpu_ids 0,1,2,3
# BraTS Sample
python main.py --config configs/brats-LBBDM-f2.yaml --sample_to_eval --gpu_ids 1