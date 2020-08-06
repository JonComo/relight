import torch
import torch.nn as nn
# import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, ), std=(0.5, )),
])

def imshow(img):
    # npimg in (channel, height, width)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Add file name to the label, can be iterated as follow:
# for i, data in enumerate(train_loader):
#     images,labels,paths = data
#     paths = [path.split('/')[-1] for path in paths]
#     print(paths)
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)

train_dataset = ImageFolderWithPaths(
    root='./dataset/',
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    num_workers=5,
    shuffle=True,
    drop_last=True
)

img_shape = (1, 224, 224)
cuda = True if torch.cuda.is_available() else False

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            # Pixel Coord + Light Coord + Average Pixel Color
            nn.Linear(2 + 2 + 1, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            # TODO: ADD DROPOUT
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, pixel_coord, light_coord, average_rgb):
        d_in = torch.cat((pixel_coord, light_coord, average_rgb), -1)
        gray = self.model(d_in)
        return gray

# Loss functions
loss = torch.nn.MSELoss()

# Initialize generator and discriminator
DNN = NN()

if cuda:
    DNN.cuda()
    loss.cuda()

# Optimizers
optimizer = torch.optim.Adam(DNN.parameters(), lr=0.002, betas=(0.9, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
CharTensor = torch.cuda.CharTensor if cuda else torch.CharTensor

loaded_avg_img = np.load('avg_img.npy')
img_shape = (1, 224, 224)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    width = img_shape[1]
    height = img_shape[2]
    avg_img = torch.from_numpy(loaded_avg_img).to(device)
    avg_img = Variable(avg_img.type(FloatTensor))
    img_tensor = torch.FloatTensor().to(device)
    for k in range(height):
        for j in range(width):
            # Pixel Coord + Light Coord + Average Pixel Color
            avg_value = Variable(avg_img[j + k * width].repeat(1, n_row).view(n_row, 1).type(FloatTensor))
            pixel_coord = Variable(torch.FloatTensor([j, k])).repeat(1, n_row).view(n_row, 2).to(device)
            # Default sample location is the light in the middle
            light_coord = Variable(torch.FloatTensor([0.0, 0.0])).repeat(1, n_row).view(n_row, 2).to(device)
            generate_pixel = DNN(pixel_coord, light_coord, avg_value)
            img_tensor = torch.cat((img_tensor, generate_pixel), 1)
    
    imgs = img_tensor.view(img_tensor.size(0), *img_shape)
    save_image(imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

epochs = 20
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        # we are using paths here instead of labels
        imgs,labels,paths = data
        paths = np.array([path.split('/')[-1][:-4].split('_') for path in paths]).astype(float)
        positions = torch.from_numpy(paths).to(device)
        avg_img = torch.from_numpy(loaded_avg_img).to(device)
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]
        width = imgs.shape[2]
        height = imgs.shape[3]

        # Configure input
        # Pixel Coord + Light Coord + Average Pixel Color
        avg_img = Variable(avg_img.type(FloatTensor))
        light_coord = Variable(positions.type(FloatTensor))
#         print(avg_img.shape) 
#         print(light_coord.shape)
        total_losses = 0
        for k in range(height):
            for j in range(width):
                optimizer.zero_grad()
                # Process the batch at this pixel coord
                pixel_value = Variable(imgs[:, 0, j, k].view(20, 1).type(FloatTensor))
                avg_value = Variable(avg_img[j + k * width].repeat(1, 20).view(20, 1).type(FloatTensor))
                pixel_coord = Variable(torch.FloatTensor([j, k])).repeat(1, 20).view(20, 2).to(device)
                generate_pixel = DNN(pixel_coord, light_coord, avg_value)
                # Image with all lighting conditions
                total_loss = loss(generate_pixel, pixel_value)
                total_losses += total_loss.item()
                if j == width/2 and k == width/2:
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [Pixel %d-%d] [Loss: %f]"
                    % (epoch + 1, epochs, i + 1, len(train_loader), j, k, total_loss)
                    )
                total_loss.backward()
                optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Total loss: %f]"
            % (epoch + 1, epochs, i + 1, len(train_loader), total_losses)
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % 1 == 0:
            sample_image(n_row=20, batches_done=batches_done)
