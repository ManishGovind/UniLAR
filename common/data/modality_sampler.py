import random
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from torch.utils.data import DataLoader, ConcatDataset


from torch.utils.data import Sampler
import random


    

# class ConcatBalancedCollator:
#     def __init__(self, dataset, split_index):
#         self.dataset = dataset
#         self.split_index = split_index  

#     def __call__(self, batch):
#         rgb_batch = [item for item in batch if item["idx"] < self.split_index]
#         depth_batch = [item for item in batch if item["idx"] >= self.split_index]
        
#         print(len(rgb_batch) , len(depth_batch))
#         selected = rgb_batch + depth_batch
        
#         return selected
        
    
class RandomModalityIterator:
    def __init__(self, dl_rgb, dl_depth, total_batches):
        self.it_rgb = iter(dl_rgb)
        self.it_depth = iter(dl_depth)
        self.total_batches = total_batches
        self.counter = 0

    def __next__(self):
        if self.counter >= self.total_batches:
            raise StopIteration

        modality = "rgb" if self.counter % 2 == 0 else "depth"
        self.counter += 1

        try:
            if modality == "rgb":
                return next(self.it_rgb)
            else:
                return next(self.it_depth)
        except StopIteration:
            raise StopIteration  


class RandomModalityBatchDataloader:
    def __init__(self, dataloader_rgb, dataloader_depth):
        self.dl_rgb = dataloader_rgb
        self.dl_depth = dataloader_depth
        self.dataset = ConcatDataset([self.dl_rgb.dataset, self.dl_depth.dataset])
        self.total_batches = min(len(self.dl_rgb), len(self.dl_depth)) * 2  

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        return RandomModalityIterator(self.dl_rgb, self.dl_depth, self.total_batches)







# class RandomModalityBatchDataloader:
#     def __init__(self, dataloader_rgb, dataloader_depth):
#         self.dl_rgb = dataloader_rgb
#         self.dl_depth = dataloader_depth
#         self.dataset = ConcatDataset([self.dl_rgb.dataset, self.dl_depth.dataset])
#         self.total_batches = min(len(self.dl_rgb), len(self.dl_depth)) * 2  
#         self._reset_iters()

#     def _reset_iters(self):
#         self.it_rgb = iter(self.dl_rgb)
#         self.it_depth = iter(self.dl_depth)
#         self.counter = 0  

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.counter >= self.total_batches:
#             raise StopIteration

#         modality = "rgb" if self.counter % 2 == 0 else "depth"
#         self.counter += 1

#         try:
#             if modality == "rgb":
#                 batch = next(self.it_rgb)
#             else:
#                 batch = next(self.it_depth)
#         except StopIteration:
#             self._reset_iters()
#             if modality == "rgb":
#                 batch = None
#                 self.it_rgb = iter(self.dl_rgb) 
#             else:
#                 batch = None
#                 self.it_depth = iter(self.it_depth)     
#                 raise StopIteration
#         return batch

#     def __len__(self):
#         return self.total_batches 
