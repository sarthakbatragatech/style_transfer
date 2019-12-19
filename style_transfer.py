from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
# efficient gradient descents
import torch.optim as optim 
from PIL import Image
import matplotlib.pyplot as plt
# transform PIL images into tensors
import torchvision.transforms as transforms 
import torchvision.models as models
#to deep copy the models
import copy
import os
from utils import image_loader, imshow
from utils import ContentLoss, StyleLoss, Normalization

def run_model(content, style, dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Use smaller image size if gpu isn't available
    if torch.cuda.is_available():
        imsize = (512, 512) 
    else:
        # Kept it low for quick debugging
        imsize = (32, 32)  

    # Resize image and transform to torch tensor
    tfms = [
        transforms.Resize(imsize),
        transforms.ToTensor()
    ]
    loader = transforms.Compose(tfms)

    style_img = image_loader(dir + '/' + content, loader, device)
    content_img = image_loader(dir + '/' + style, loader, device)

    unloader = transforms.ToPILImage()
    
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
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

    input_img = content_img.clone()
    # For white noise, uncomment the line below
    # input_img = torch.randn(content_img.data.size(), device=device)

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
        num_steps=50, # Kept it low for quick debugging
        style_weight=1000000,
        content_weight=1):
            
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
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
                    # plt.figure()
                    # imshow(input_img)
                    # plt.show()

                return style_score + content_score

            optimizer.step(closure)

        # Clamp the data one last time
        input_img.data.clamp_(0, 1)

        return input_img

    output = run_style_transfer(
        cnn,
        vgg_mean,
        vgg_std,
        content_img,
        style_img,
        input_img)

    plt.figure()
    output_name = imshow(
        output, 
        loader,
        unloader,
        dir,
        title='Output Image',
        output=True)

    return output_name
