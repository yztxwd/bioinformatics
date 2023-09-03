from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import user_defined_dataset as transformed_dataset

# dataloader example instead of single input
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

# original model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = resnet18(pretrained=True)
model = model.to(device)

# a dict to store the activations
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

# register forward hooks on the layers of choice
h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))
h2 = model.maxpool.register_forward_hook(getActivation('maxpool'))
h3 = model.layer3[0].downsample[1].register_forward_hook(getActivation('comp'))

avgpool_list, maxpool_list, comp_list = [], [], []
# go through all the batches in the dataset
for X, y in dataloader:
  # forward pass -- getting the outputs
  out = model(X)
  # collect the activations in the correct list
  avgpool_list.append(activation['avgpool']
  maxpool_list.append(activation['maxpool']
  comp_list.append(activation['comp']
	
# detach the hooks
h1.remove()
h2.remove()
h3.remove()
