from fastai.basic_train import Learner

def train_fastai(data, model, epochs, learning_rate)
  learn = Learner(db, model, loss_func=nn.CrossEntropyLoss())
  learn.fit(epochs, learning_rate)
  return model
