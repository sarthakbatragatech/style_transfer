import torch
import matplotlib.pyplot as plt
# transform PIL images into tensors
import torchvision.transforms as transforms 
import torchvision.models as models
from utils import image_loader, imshow, get_style_model_and_losses
from utils import get_input_optimizer, run_style_transfer
from utils import gram_matrix, ContentLoss, StyleLoss, Normalization

def run_model(content, style, folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use smaller image size if gpu isn't available
    if torch.cuda.is_available():
        imsize = (512, 512) 
    else:
        # Kept it small for quick debugging
        imsize = (128, 128)  

    # Resize image and transform to torch tensor
    tfms = [transforms.Resize(imsize), transforms.ToTensor()]
    loader = transforms.Compose(tfms)

    style_img = image_loader(folder + '/' + content, loader, device)
    content_img = image_loader(folder + '/' + style, loader, device)

    unloader = transforms.ToPILImage()
    
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    input_img = content_img.clone()
    # For white noise, uncomment the line below
    # input_img = torch.randn(content_img.data.size(), device=device)

    output = run_style_transfer(cnn, vgg_mean, vgg_std, content_img, 
                                style_img, input_img, device, loader, 
                                unloader, folder)

    plt.figure(figsize=(8,8))
    output_name = imshow(output, loader, unloader, folder, 
                        title='Output Image', output=True)

    return output_name