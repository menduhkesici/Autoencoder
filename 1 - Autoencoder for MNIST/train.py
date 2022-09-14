import os

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # input: N * 1 * 28 * 28
            nn.Conv2d(1, 6, 3, stride=1, padding=1),  # output: N * 6 * 28 * 28
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # output: N * 6 * 14 * 14
            nn.Conv2d(6, 12, 3, stride=1, padding=1),  # output: N * 12 * 14 * 14
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # output: N * 12 * 7 * 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(12 * 7 * 7, 128),  # output: N * 128
            nn.ReLU(True),
            nn.Linear(128, 10)  # output: N * 10
        )
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
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        return x


def train(continue_from_last=False, num_of_epochs=10, filename=None, train_on_gpu=False):
    # TRAIN PARAMETERS #
    batch_size = 64
    learning_rate = 1e-3
    tests_per_epoch = 3
    tests_per_image = 64  # should be equal to or less than batch size
    # ---------------- #

    if os.path.isdir('./dc_output') and continue_from_last is False:
        while len(os.listdir('./dc_output')) > 0:
            print('dc_output folder is not empty.')
            reply = str(input('Do you want to continue from last epoch (y/n): ')).lower().strip()
            if reply[0] == 'y':
                continue_from_last = True
                break
            elif reply[0] == 'n':
                break

    device = torch.device("cuda:0" if (torch.cuda.is_available() and train_on_gpu) else "cpu")
    print("Device is", device)

    model = Autoencoder().to(device)
    start_epoch = 0
    time_logger = []
    train_losslogger = []
    test_losslogger = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if continue_from_last:
        if filename is None:
            files = [file for file in os.listdir('./dc_output') if file.endswith(".pth")]
            max_len = len(max(files, key=len))
            filename = max([file for file in files if len(file) == max_len])
        checkpoint = torch.load(os.path.join('./dc_output', filename))
        start_epoch = checkpoint['epoch']
        num_of_epochs = start_epoch + num_of_epochs
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        time_logger = checkpoint['time_logger']
        train_losslogger = checkpoint['train_losslogger']
        test_losslogger = checkpoint['test_losslogger']
        print("=> loaded checkpoint '{}'".format(filename))

    # print(model)

    # Configures the model for training
    model.train()

    # Training dataset
    train_data = torchvision.datasets.MNIST(
        root='./',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    # Test dataset
    test_data = torchvision.datasets.MNIST(
        root='./',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    for epoch in range(start_epoch, num_of_epochs):
        time.sleep(0.1)
        print('epoch [{}/{}] started.'.format(epoch + 1, num_of_epochs))
        time.sleep(0.1)
        total_train_loss, total_train_no = 0, 0
        for train_no, (train_inputs, train_labels) in enumerate(tqdm(train_loader)):
            img = train_inputs.to(device)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss) * img.size()[0]
            total_train_no += img.size()[0]
            if train_no % (len(train_loader) // tests_per_epoch) == (len(train_loader) // tests_per_epoch) - 1:
                total_test_loss, total_test_no = 0, 0
                test_image = None
                for test_no, (test_inputs, test_labels) in enumerate(test_loader):
                    img = test_inputs.to(device)
                    output = model(img)
                    test_loss = criterion(output, img)
                    total_test_loss += float(test_loss) * img.size()[0]
                    total_test_no += img.size()[0]
                    if test_no == 0:
                        test_image = torch.cat([img.data[:tests_per_image], output.data[:tests_per_image]], dim=0)
                if not os.path.isdir('./dc_output'):
                    os.mkdir('./dc_output')
                train_losslogger.append(total_train_loss / total_train_no)
                test_losslogger.append(total_test_loss / total_test_no)
                time_logger.append(str(epoch + 1) + '_' + str(train_no // (len(train_loader) // tests_per_epoch)))
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'time_logger': time_logger,
                         'train_losslogger': train_losslogger, 'test_losslogger': test_losslogger, }
                torch.save(state, './dc_output/conv_autoencoder_{}_{}.pth'
                           .format(epoch + 1, train_no // (len(train_loader) // tests_per_epoch)))
                save_image(test_image, './dc_output/test_{}_{}_train_loss_{:.4f}_test_loss_{:.4f}.jpg'
                           .format(epoch + 1, train_no // (len(train_loader) // tests_per_epoch),
                                   total_train_loss / total_train_no, total_test_loss / total_test_no),
                           nrow=tests_per_image)
                total_train_loss, total_train_no = 0, 0

    print("Training completed with {} epochs.".format(num_of_epochs))


def plot_loss(filename=None):
    if filename is None:
        files = [file for file in os.listdir('./dc_output') if file.endswith(".pth")]
        max_len = len(max(files, key=len))
        filename = max([file for file in files if len(file) == max_len])
    checkpoint = torch.load(os.path.join('./dc_output', filename))
    time_logger = checkpoint['time_logger']
    train_losslogger = checkpoint['train_losslogger']
    test_losslogger = checkpoint['test_losslogger']
    plt.plot(list(range(len(time_logger))), train_losslogger, 'g', label='Train loss')
    plt.plot(list(range(len(time_logger))), test_losslogger, 'r', label='Test loss')
    plt.xticks(np.arange(len(time_logger)), time_logger, rotation=75)
    min_y = round(min(min(train_losslogger, test_losslogger)), 2) - 0.01
    max_y = round(max(max(train_losslogger, test_losslogger)), 2) + 0.01
    plt.yticks(np.arange(min_y, max_y, 0.01))
    plt.legend()
    plt.grid(linestyle='--', linewidth='0.5', alpha=0.7)
    print("=> plotted checkpoint '{}')".format(filename))
    plt.show()


if __name__ == '__main__':
    train(continue_from_last=True, num_of_epochs=5, train_on_gpu=True)
    plot_loss()
    pass
