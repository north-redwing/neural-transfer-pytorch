import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from loss import ContentLoss, StyleLoss


def image_loader(image_name, imsize=(128,128), device=torch.device('cpu')):
    loader_mono = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    loader_rgb = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
    ])
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimension
    if image.mode == 'RGB':
        image = loader_rgb(image).unsqueeze(0)
    else:
        image = loader_mono(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class Normalization(nn.Module):
    """
    VGG networks are trained on images with each channel normalized
    by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    We will use them to normalize the image before sending it into the network.
    """

    def __init__(self, device=torch.device('cpu')):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(
            device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, style_img, content_img,
                               device=torch.device('cpu')):
    """
    We need to add our content loss and style loss layers immediately
    after the convolution layer they are detecting.
    To do this we must create a new Sequential module
    that has content loss and style loss modules correctly inserted.
    """

    # desired depth layers to compute style/content losses:
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = copy.deepcopy(cnn)
    normalization = Normalization().to(device)

    content_losses, style_losses = [], []

    model = nn.Sequential(normalization)

    n_conv = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            n_conv += 1
            name = 'conv_{}'.format(n_conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(n_conv)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(n_conv)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(n_conv)
        else:
            raise RuntimeError(
                'Unrecognized layer:{}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss
            target_feature = model(content_img).detach()
            content_loss = ContentLoss(target_feature)
            model.add_module('content_loss_{}'.format(n_conv), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(n_conv), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    """
    As Leon Gatys, the author of the algorithm,
    we will use L-BFGS algorithm to run our gradient descent.
    """
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
