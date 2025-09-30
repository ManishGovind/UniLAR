# from PIL import Image
# import requests
# from transformers import AutoProcessor, SiglipVisionModel

# model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-224")
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-224")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooled_output = outputs.pooler_output  # pooled features

# print(last_hidden_state.shape , pooled_output.shape)


import torch.nn as nn
from transformers import SiglipVisionModel
from PIL import Image

class SiglipEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path="google/siglip-so400m-patch14-224"):
        super().__init__()
        print("Loading Siglip Model")
        self.model = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path)
        
    def forward(self, images):
        """
        Args:
            images: torch.FloatTensor [B, 3, H, W], already preprocessed
        Returns:
            patch_features: [B, seq_len, hidden_dim]
        """
        # print(images.shape)
        outputs = self.model(images)
        patch_features = outputs.last_hidden_state   # [B, seq_len, hidden_dim]
        return None, patch_features

