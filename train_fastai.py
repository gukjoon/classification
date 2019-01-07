from fastai.basic_train import Learner, LearnerCallback
from torch.utils.data import DataLoader
from fastai.basic_data import DataBunch
from fastai.metrics import accuracy
from fastai.vision import Image as FImage
from fastai.vision.transform import get_transforms
from fastai.callbacks.hooks import HookCallback
import torch.nn as nn

tt = get_transforms()[0]

def apply_tfms(data):
    (imgs, labels) = data
    app = []
    lab = []
    for i in range(0, 4):
        for img in imgs:
            nimg = FImage(img)
            app.append(nimg.apply_tfms(tt, size=224)._px)
        lab.append(labels)
    return (torch.stack(app).cuda(), torch.cat(lab).cuda())

class KeepAliveLogger(LearnerCallback):
  def __init__(self, learn):
      super().__init__(learn)

  def on_batch_end(self, train, **kwargs):
    print(kwargs)

def train_fastai(training_data, validation_data, model, epochs, learning_rate):
  db = DataBunch(
    DataLoader(training_data, batch_size=32), 
    DataLoader(validation_data, batch_size=32),
    tfms=[apply_tfms],
    device='cpu'
  )
  db.valid_dl.tfms = None
  db.valid_dl.device = 'cuda'
  learn = Learner(db, model, loss_func=nn.NLLLoss(), metrics=[accuracy])
  learn.precompute = False
  learn.model.cuda()
  learn.fit_one_cycle(epochs, learning_rate, callbacks=[KeepAliveLogger])
  return model
