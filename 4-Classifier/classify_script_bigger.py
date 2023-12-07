'''
Script to load model and make prediction on one image.

args:
image_path = path to image on disk
model_number = [18,34,50,101,152]
    the model number of resnet to be loaded and used
    for classification

'''

import torch    
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image
## from PyTorch documentation
import torch
import urllib



def predict_one_image(image_path,model_number,classes_filename="imagenet_classes.txt"):

    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if model_number == '18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    elif model_number == '34':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    elif model_number == '50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    elif model_number == '101':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    elif model_number == '152':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    
    
    model.eval()



    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    with open(classes_filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())







def main():
    parser = argparse.ArgumentParser(description='predict on one image')
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('model_number', type=str, help='resnet model to use (18,34,50,101,152)')
    args = parser.parse_args()

    predict_one_image(args.image_path, args.model_number)
    

if __name__ == '__main__':
    main()