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
        self.encoder = nn.Sequential(  # input: N * 3 * 32 * 32
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # output: N * 8 * 32 * 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # output: N * 8 * 16 * 16
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # output: N * 16 * 16 * 16
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # output: N * 12 * 8 * 8
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # output: N * 32 * 8 * 8
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # output: N * 32 * 4 * 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 4 * 4, 128),  # output: N * 128
            nn.ReLU(True),
            nn.Linear(128, 32)  # output: N * 32
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 1024, 4, 1, 0, bias=False),  # output: N * 1024 * 4 * 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # output: N * 512 * 8 * 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # output: N * 256 * 16 * 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, 4, 2, 1, bias=False),  # output: N * 3 * 32 * 32
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
    num_of_workers = 1
    normalize_values_mean = (0.4914, 0.4822, 0.4465)
    normalize_values_std = (0.2023, 0.1994, 0.2010)
    learning_rate = 1e-3
    tests_per_epoch = 3
    tests_per_image = 64  # should be equal to or less than batch_size
    tests_per_row = 16  # should be a dividend of tests_per_image
    # ---------------- #

    assert tests_per_image <= batch_size, 'tests_per_image should be equal to or less than batch_size.'
    assert tests_per_row > 0, 'tests_per_rows must be greater than zero.'
    assert tests_per_image % tests_per_row == 0, 'tests_per_rows should be a dividend of tests_per_image.'
    if os.path.isdir('./dc_output') and continue_from_last is False:
        assert len(os.listdir('./dc_output')) == 0, 'dc_output folder is not empty, may be overwritten.'

    # noinspection PyUnresolvedReferences
    device = torch.device("cuda:0" if (torch.cuda.is_available() and train_on_gpu) else "cpu")
    print("Device is", device)

    model = Autoencoder().to(device, non_blocking=True)
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
        model.to(device, non_blocking=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        time_logger = checkpoint['time_logger']
        train_losslogger = checkpoint['train_losslogger']
        test_losslogger = checkpoint['test_losslogger']
        print("=> loaded checkpoint '{}'".format(filename))

    # print(model)

    # Configures the model for training
    model.train()

    # Training dataset
    train_data = torchvision.datasets.CIFAR10(
        root='./',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_values_mean,
                                 std=normalize_values_std)
        ]))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_of_workers,
        pin_memory=True)

    # Test dataset
    test_data = torchvision.datasets.CIFAR10(
        root='./',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_values_mean,
                                 std=normalize_values_std)
        ]))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_of_workers,
        pin_memory=True)

    for epoch in range(start_epoch, num_of_epochs):
        time.sleep(0.1)
        print('epoch [{}/{}] started.'.format(epoch + 1, num_of_epochs))
        time.sleep(0.1)
        total_train_loss, total_train_no = 0, 0
        for train_no, (train_inputs, train_labels) in enumerate(tqdm(train_loader)):
            img = train_inputs.to(device, non_blocking=True)
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
                    img = test_inputs.to(device, non_blocking=True)
                    output = model(img)
                    test_loss = criterion(output, img)
                    total_test_loss += float(test_loss) * img.size()[0]
                    total_test_no += img.size()[0]
                    if test_no == 0:
                        test_image = torch.cat((img.data[:tests_per_row], output.data[:tests_per_row]), dim=0)
                        for i in range(1, tests_per_image // tests_per_row):
                            test_image = torch.cat((test_image,
                                                    img.data[i*tests_per_row:(i+1)*tests_per_row],
                                                    output.data[i*tests_per_row:(i+1)*tests_per_row]), dim=0)
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
                           nrow=tests_per_row, normalize=True)
                total_train_loss, total_train_no = 0, 0

    print("Training completed with {} epochs.".format(num_of_epochs))


def plot_loss(filename=None, decimals=2):
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
    plt.xticks(np.arange(len(time_logger)), time_logger, rotation=90)
    min_y = round(min(min(train_losslogger, test_losslogger)), int(decimals)) - 10**(-decimals)
    max_y = round(max(max(train_losslogger, test_losslogger)), int(decimals)) + 10**(-decimals)
    plt.yticks(np.arange(min_y, max_y, 10**(-decimals)))
    plt.legend()
    plt.grid(linestyle='--', linewidth='0.5', alpha=0.7)
    print("=> plotted checkpoint '{}'".format(filename))
    plt.show()


if __name__ == '__main__':
    train(continue_from_last=False, num_of_epochs=10, train_on_gpu=True)
    plot_loss()
    pass
