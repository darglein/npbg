import os

import cv2
import torch
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from npbg.criterions.vgg_loss import PrintTensorInfo


def ClassfifyImage(input_image):
    global model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)


    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

def run(input_dir, output_dir):
    outfiles = []
    for filename in os.listdir(input_dir):

        in_file = input_dir + filename
        out_file = output_dir + filename + "_class_color.png"
        out_file2 = output_dir + filename + "_class.png"

        outfiles.append(filename + "_class.png")
        continue
        input_image = Image.open(in_file)
        output_predictions = ClassfifyImage(input_image)
        output_predictions  = output_predictions.byte()
        PrintTensorInfo(output_predictions)

        trans = transforms.ToPILImage()
        img = trans(output_predictions.cpu())
        img.save(out_file2)




        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        print("save to ", out_file)
        r.save(out_file)

        # print("show")
        # plt.imshow(r)
        # plt.waitforbuttonpress()

    f = open(output_dir + "classes.txt", "w")
    for a in outfiles:
        f.write(a + "\n")
    f.close()

if __name__ == '__main__':
    print("debug")
    global model
    model = torch.hub.load('pytorch/vision:v0.4.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    run("/HD/tank_and_temples/reconstructions/Train/images/", "/HD/tank_and_temples/reconstructions/Train/classes/")