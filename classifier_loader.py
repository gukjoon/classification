from PIL import ImageMath
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class ClassifierLoader:
  def __init__(self, image_generator, label_generator, label_map, transforms_in):
    self.image_gen = list(image_generator)
    
    self.label_gen = [i.result.read().strip().decode('utf-8') for i in label_generator]
    self.label_map = label_map
        
    # analyze labels    
    print('isize', len(self.image_gen))
    print('lsize', len(self.label_gen))
    
    self.transform = transforms.Compose(transforms_in)

  def __getitem__(self, index):
    image = Image.open(self.image_gen[index].result)
    label = self.label_map[self.label_gen[index]]
    return (self.transform(image), label)

  def __len__(self):
    return len(self.image_gen)

def classifier_loader(images_generator, labels_generator, label_map):
  dataset = ClassifierLoader(images_generator, labels_generator, label_map, [
    transforms.Resize((224,224)),
    transforms.ToTensor()
  ])

  return dataset
