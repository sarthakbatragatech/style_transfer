from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# efficient gradient descents
import torch.optim as optim
# Prevent browser caching of generated image
import time
#to deep copy the models
import copy
import os

# Function to check if uploaded file has an acceptable extension
def is_file_allowed(filename):
    if not "." in filename:
        return False
    
    suffix = filename.rsplit('.', 1)[1]

    if suffix.lower() in ['jpeg', 'jpg', 'png']:
        return True
    else:
        return False

# Function to add time component to an image name
def add_time(filename):
    img_name = filename.rsplit('.')[0]
    img_suffix = filename.rsplit('.')[1]
    filename = str(time.time()).replace('.','_') +'.'+img_suffix
    return filename

# Convert image to torch tensor
def image_loader(img_path, loader, device):
    img = Image.open(img_path)
    # If PNG file, get rid of 4th channel (alpha) by converting image to JPG
    if img_path[-3:].lower() == 'png':
        img = img.convert('RGB')
    # Insert 1 in shape of the tensor at axis 0 (batch size)
    # Extra dimension is required to fit the network's input dimensions
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)

def imshow(tensor, loader, unloader, folder='', title=None, output=False):
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
            folder + '/' + output_name, 
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
                            device,
                            style_img, 
                            content_img,
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

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(
    cnn, 
    normalization_mean, 
    normalization_std,
    content_img,
    style_img,
    input_img,
    device,
    loader,
    unloader,
    folder,
    num_steps=300, # Kept it low for quick debugging
    style_weight=1000000,
    content_weight=1):
        
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, device, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    print()
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Clamp the image tensor to (0,1) range
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'Run {run[0]}:')
                print(f'Style Loss : {style_score.item():4f}')
                print(f'Content Loss: {content_score.item():4f}')
                print()
                plt.figure(figsize=(8,8))
                title = f'Run {run[0]} Image'
                imshow(input_img, loader, unloader, folder, 
                        title=title, output=True)                
            return style_score + content_score

        optimizer.step(closure)

    # Clamp the data one last time
    input_img.data.clamp_(0, 1)

    return input_img