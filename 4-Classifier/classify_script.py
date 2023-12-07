#  Python script made to make a prediction via model on the class
#       of an image.  image and 
#
#
#
#
#


import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import argparse
from PIL import Image

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


## Basic Model from Documentation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_params(path,model):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def predict_one_image(image_path,model_path,classes=classes):

    image = Image.open(image_path)

    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Preprocess Transformations
    # These should match the transforms in the training of the model
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to match the input size of your model
    transforms.ToTensor(),        
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image
        ])

    # Apply the transformations to preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add a batch dimension (batch_size=1)

    

    # Make predictions
    with torch.no_grad():
        outputs = model(input_image)

    # Get the predicted class
    _, predicted = torch.max(outputs, 1)

    print("Predicted class:", classes[predicted.item()])
    return classes[predicted.item()]



def main():
    parser = argparse.ArgumentParser(description='predict on one image')
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('model_path', type=str, help='path to trained model state dict')
    args = parser.parse_args()

    predicted_class = predict_one_image(args.image_path, args.model_path)
    return predicted_class

if __name__ == '__main__':
    main()