
def num_children(m):
    return len(list(m.children()))

flatten_model = lambda m: sum(map(flatten_model,m.children()),[]) if num_children(m) else [m]

def requires_grad(m, b):
    "If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param."
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b

def freeze_model(model, freeze_to):
  for i in flatten_model(model)[:-freeze_to]:
    requires_grad(i, False)
  for i in flatten_model(model)[-freeze_to:]:
    requires_grad(i, True)
  return model
