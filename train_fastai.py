from fastai.basic_train import Learner
from torch.utils.data import DataLoader
from fastai.basic_data import DataBunch

def train_fastai(training_data, validation_data, model, epochs, learning_rate)
  db = DataBunch(
    DataLoader(training_data, batch_size=8), 
    DataLoader(validation_data, batch_size=8)
  )
  learn = Learner(db, model, loss_func=nn.CrossEntropyLoss())
  learn.fit(epochs, learning_rate)
  return model