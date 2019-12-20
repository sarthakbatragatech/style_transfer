from PIL import Image
import torch
import matplotlib.pyplot as plt
# Prevent browser caching of generated image
import time
import torch.nn as nn
import torch.nn.functional as F

# Function to check if uploaded file has an acceptable extension
def is_file_allowed(filename):
    if not "." in filename:
        return False
    
    suffix = filename.rsplit('.', 1)[1]

    if suffix.lower() in ['jpeg', 'jpg', 'png']:
        return True
    else:
        return False

# Convert image to torch tensor
def image_loader(img_path, loader, device):
    img = Image.open(img_path)
    print(img_path)
    # If PNG file, get rid of 4th channel (alpha) by converting image to JPG
    if img_path[-3:].lower() == 'png':
        img = img.convert('RGB')
    # Insert 1 in shape of the tensor at axis 0 (batch size)
    # Extra dimension is required to fit the network's input dimensions
    img = loader(img).unsqueeze(0)
    print(img.size())
    return img.to(device, torch.float)

def imshow(tensor, loader, unloader, dir, title=None, output=False):
    # Clone the tensor so it's not changed in-place
    image = tensor.cpu().clone()
    # Removed the extra dimension added previously
    image = image.squeeze(0)      
    image = unloader(image)
    # Now we have a normal image, let's display it
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    if output:
        output_name = 'result' + '?' + str(time.time()) + '.png'
        plt.savefig(
            dir + '/' + output_name, 
            dpi=100, 
            bbox_inches=None,
            pad_inches=0.)
        plt.close()
        return output_name

class ContentLoss(nn.Module):

    def __init__(self, target):
        # Sub-class this class
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    # a is batch size, equal to 1
    # b is the number of feature maps
    # (c,d) are dimensions of feature map
    a, b, c, d = input.size()  

    # resize matrix to [b,(c*d)] form
    features = input.view(a * b, c * d)  
    
    # Compute the Gram-Matrix and normalize it
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        # Sub-class this class
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        # Sub-class this class
        super(Normalization, self).__init__()
        
        # Use .view to change the shape of the mean and std
        # tensors. They take the form [num_channels x 1 x 1] 
        # and become compatible with the image tensors
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize the image with VGG stats
        return (img - self.mean) / self.std