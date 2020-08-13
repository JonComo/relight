import torch
import torch.nn as nn
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
import copy
import torch.multiprocessing as mp

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

im_s = 32

transform = transforms.Compose([
    transforms.Resize(im_s),
    transforms.CenterCrop(im_s),
    #     transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Add file name to the label, can be iterated as follow:
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)

train_dataset = ImageFolderWithPaths(
    root='./dataset/',
    transform=transform
)
batch_size = 120
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    drop_last=True
)

img_shape = (3, im_s, im_s)
cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
CharTensor = torch.cuda.CharTensor if cuda else torch.CharTensor

# Loss functions
loss = torch.nn.MSELoss()

if cuda:
    loss.cuda()

loaded_avg_img = np.load('./avg/avg_basketball_32.npy')

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            # Pixel Coord + Light Coord + Average Pixel Color
            nn.Linear(2 + 2 + 3, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
            nn.Tanh()
        )
    def forward(self, pixel_coord, light_coord, average_rgb):
        d_in = torch.cat((pixel_coord, light_coord, average_rgb), -1)
        rgb = self.model(d_in)
        return rgb

def reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def train(model, j, k, optimizer):
    model.train()
#     epochs = 100
#     for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # we are using paths here instead of labels
        imgs,labels,paths = data
        paths = np.array([path.split('/')[-1][:-4].split('_') for path in paths]).astype(float)
        paths[:, 0] /= 10
        paths[:, 1] -= 10
        paths[:, 1] /= 10
        print(paths)
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
        total_losses = 0

        pixel_value = Variable(imgs[:, :, j, k].view(batch_size, 3).type(FloatTensor))

        avg_value = Variable(avg_img[j + k * width].repeat(1, batch_size).view(batch_size, 3).type(FloatTensor))

        pixel_coord = Variable(torch.FloatTensor([j/width, k/height])).repeat(1, batch_size).view(batch_size, 2).to(device)

        generate_pixel = model(pixel_coord, light_coord, avg_value)

        # Image with all lighting conditions
        total_loss = loss(generate_pixel, pixel_value)
        # total_losses += total_loss.item()
        total_loss.backward()
        optimizer.step()

def save_model(model, path):
    checkpoint = {'model': model,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer_1.state_dict()}
    torch.save(checkpoint, path)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print('GPU Spawning')
    except RuntimeError:
        print('Failed to use GPU')
        pass
    try:
        mp.set_sharing_strategy('file_system')
        print('File System')
    except RuntimeError:
        print('File Descriptor')
        pass

    num_processes = 1
    model_1 = NN().to(device)
    # model_2 = copy.deepcopy(model_1)
    # model_3 = copy.deepcopy(model_1)
    # model_4 = copy.deepcopy(model_1)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.2, betas=(0.9, 0.999))
    # optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.002, betas=(0.9, 0.999))
    # optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=0.002, betas=(0.9, 0.999))
    # optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=0.002, betas=(0.9, 0.999))
    processes = []

    # models = [model_1, model_2, model_3, model_4]
    models = [model_1]
    # optimizers = [optimizer_1, optimizer_2, optimizer_3, optimizer_4]
    optimizers = [optimizer_1]
    model_1.share_memory()

    save_model(model_1, './check_pt/before.pth')

    # for i in range(num_processes):
    #     optimizers[i].zero_grad()

    for j in range(1):
        for k in range(1):
            for i in range(num_processes):
                p = mp.Process(target=train, args=(models[i], j, k, optimizers[i]))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

    # sub_grad = []

    # for i in range(1, num_processes):
    #     j = 0
    #     print(sub_grad)
    #     if i == 1:
    #         for p in models[i].parameters():
    #             sub_grad += [p.grad]
    #     else:
    #         for p in models[i].parameters():
    #             sub_grad[j] += p.grad
    #             j += 1
    #
    # j = 0
    # for p in models[0].parameters():
    #     p.grad += sub_grad[j]
    #     j += 1

    # optimizer_1.step()

    save_model(model_1, './check_pt/after.pth')

