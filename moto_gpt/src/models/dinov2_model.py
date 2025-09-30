# from PIL import Image
# import requests
# from transformers import AutoImageProcessor, AutoModel

# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# model = AutoModel.from_pretrained('facebook/dinov2-large')

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state

# print(last_hidden_state.shape)


import torch.nn as nn
from transformers import AutoModel
from PIL import Image

class Dinov2Encoder(nn.Module):
    def __init__(self, use_obs_feature, pretrained_model_name_or_path="facebook/dinov2-large"):
        super().__init__()
        print("Loading Dinov2 Model")
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.use_obs_feature = use_obs_feature
        
    def forward(self, images):
        """
        Args:
            images: torch.FloatTensor [B, 3, H, W], already preprocessed
        Returns:
            patch_features: [B, seq_len, hidden_dim]
        """
        # print(images.shape)
        outputs = self.model(images)
        last_hidden_states = outputs.last_hidden_state   # [B, seq_len, hidden_dim]

        obs_features = last_hidden_states[:, 0, :]
        patch_features = last_hidden_states[:, 1:, :]

        if self.use_obs_feature:
            # print(obs_features.shape ,  patch_features.shape)
            return obs_features, patch_features
             
        else:
            return None, patch_features
        

