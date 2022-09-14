import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 16, 3, 1, 0, bias=False),  # output: N * 16 * 3 * 3
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, 2, 1, bias=False),  # output: N * 8 * 7 * 7
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False),  # output: N * 4 * 14 * 14
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 4, 2, 1, bias=False),  # output: N * 1 * 28 * 28
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x


# Driver module to test the generator
if __name__ == '__main__':

    files = [file for file in os.listdir('./dc_output') if file.endswith(".pth")]
    max_len = len(max(files, key=len))
    filename = max([file for file in files if len(file) == max_len])
    state_dict = torch.load(os.path.join('./dc_output', filename))['state_dict']
    print("=> loaded checkpoint '{}'".format(filename))

    # Only take the parameters that are used in this model from the saved model
    model = Generator()
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param.data)

    # print(model)

    # Configures the model for testing
    model.eval()

    with torch.no_grad():
        while True:
            inp = torch.rand(1, 10) * random.randint(1, 3)
            inp[0][random.randint(0, 9)] = 5
            inp = Variable(inp)
            output = model(inp)
            plt.imshow(output.detach().numpy().squeeze(), cmap='gray')
            plt.show()

    pass
