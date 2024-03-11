import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


def calc_FID(input_path1, input_path2):
    fid_value = fid_score.calculate_fid_given_paths([input_path1, input_path2],
                                                    batch_size=1,
                                                    device=torch.device('cuda:0'),
                                                    dims=2048)  # 2048,768,192,64
    print('FID value:', fid_value)
    return fid_value
