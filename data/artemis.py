import os
import unicodedata
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image #compare
from PIL import Image
import emotions
import styles

from data_processors import  CaptionProcessor
from data_processors import ImageProcessor

#dist
def img_distribution(g):
    image_distribution = np.zeros(9, dtype=np.float32)
    for l in g.emotion_label:
        image_distribution[l] += 1.0
    return image_distribution/sum(image_distribution)

##max emotion
def img_max_emotion(g):
    image_distribution = np.zeros(9, dtype=np.float32)
    for l in g.emotion_label:
        image_distribution[l] += 1.0
    return np.argmax(image_distribution)


class ArtEmis(Dataset):
    def __init__(self,data_csv,img_dir):
        
        self.df = pd.read_csv(data_csv,header=0)
        print(self.df)
        self.img_dir = img_dir
        self.vis_processor=ImageProcessor()
        self.text_processor=CaptionProcessor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join("/ibex/ai/home/kimds/Research/P2/withLLM/making_it_works/data/img_after/img_"+str(idx)+".jpg")
        img_path = unicodedata.normalize('NFD', img_path)
        image = Image.open(img_path).convert("RGB")
        image = self.vis_processor(image)
#        print(image)
        caption = self.text_processor(self.df.iloc[idx,2])
        print(caption)
        return {"image":image,"text":caption}
