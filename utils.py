from PIL import Image
import torch
import matplotlib.pyplot as plt
# Prevent browser caching of generated image
import time
import torch.nn as nn
import torch.nn.functional as F
#to deep copy the models
import copy

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
    # Insert 1 in shape of the tensor at axis 0 (batch size)
    # Extra dimension is required to fit the network's input dimensions
    img = loader(img).unsqueeze(0)
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

# Calculate content loss at conv level 4
content_layers_default = ['conv_4']

# Calculate style loss at each level
style_layers_default = [
    'conv_1', 
    'conv_2', 
    'conv_3', 
    'conv_4', 
    'conv_5'
]

def get_style_model_and_losses(cnn, 
                            normalization_mean, 
                            normalization_std,
                            style_img, 
                            content_img,
                            device,
                            content_layers=content_layers_default,
                            style_layers=style_layers_default
                            ):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            # If we see a conv layer, increment i
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # Replace in-place version with out-of-place as it
            # doesn't work too well with style/content losses
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    # We then trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        # As soon as we encounter the last style/content layer, break the loop                       
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    # Grab the model until the cut-off point                           
    model = model[:(i + 1)]

    return model, style_losses, content_losses