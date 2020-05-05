import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from utils import image_loader, imshow, get_style_model_and_losses, \
    get_input_optimizer


def main():
    # params
    device = torch.device('cpu')
    num_steps = 300
    style_weight = 10
    content_weight = 1

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Style Image / Content Image
    # style_img = image_loader('./picasso.jpg')
    # content_img = image_loader('./dancing.jpg')
    style_img = image_loader('./style.jpg')
    content_img = image_loader('./content.jpg')

    # Input Image
    input_img = content_img.clone()
    plt.figure()
    imshow(input_img, title='Input Image')

    assert style_img.size() == content_img.size(), \
        'We need to import style and content images of the same size'
    # run the tansfer
    print('Building the style transfer model...')
    model, style_losses, content_losses = \
        get_style_model_and_losses(cnn, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')

    for step in range(1, num_steps, 1):

        def closure():
            # Clamp all elements in input into the range [0, 1]
            input_img.data.clamp = (0, 1)
            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for style_loss in style_losses:
                style_score += style_loss.loss
            for content_loss in content_losses:
                content_score += content_loss.loss
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            if step % 5 == 0:
                print("step {}:".format(step))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return loss.item()

        optimizer.step(closure)

    output_img = input_img.data.clamp(0, 1)

    plt.figure()
    imshow(output_img, title='Output Image')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
