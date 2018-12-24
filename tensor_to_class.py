import torch.nn as nn

def tensor_to_class(tensor, rev_map):
  idx = nn.Softmax()(tensor).argmax().data.item()
  return [rev_map[idx].encode('utf-8')]
