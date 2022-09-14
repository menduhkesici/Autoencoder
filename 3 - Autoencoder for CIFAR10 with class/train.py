import os

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import time
import math
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
        self.classifier = nn.Sequential(  # input: N * (32 * 4 * 4)
            nn.Linear(32 * 4 * 4, 64),  # output: N * 64
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 10)  # output: N * 10
        )
        self.decoder = nn.Sequential(  # input: N * 32 * 4 * 4
            nn.ConvTranspose2d(32, 1024, 3, 1, 1, bias=False),  # output: N * 1024 * 4 * 4
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        class_out = self.classifier(torch.flatten(encoded, 1))
        return decoded, class_out


def train(continue_from_last=False, num_of_epochs=10, filename=None, train_on_gpu=False):
    # TRAIN PARAMETERS #
    batch_size = 64
    num_of_workers = 1
    normalize_values_mean = (0.4914, 0.4822, 0.4465)
    normalize_values_std = (0.2023, 0.1994, 0.2010)
    learning_rate = 1e-3
    loss_ratio = 3  # loss = loss_decoder + loss_ratio * loss_class_out
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
    device = torch.device("cuda:0" if (train_on_gpu and torch.cuda.is_available()) else "cpu")
    print("Device is", device)

    model = Autoencoder().to(device, non_blocking=True)
    start_epoch = 0
    time_logger = []
    train_losslogger = []
    test_losslogger = []
    train_acclogger = []
    test_acclogger = []
    criterion_decoder = nn.MSELoss()
    criterion_classifier = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if continue_from_last:
        assert os.path.isdir('./dc_output'), 'The folder "dc_output" is not found.'
        if filename is None:
            files = [file for file in os.listdir('./dc_output') if file.endswith(".pth")]
            assert len(files) > 0, 'There is no checkpoint to load from.'
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
        train_acclogger = checkpoint['train_acclogger']
        test_acclogger = checkpoint['test_acclogger']
        print("=> loaded checkpoint '{}'".format(filename))

    # print(model)

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

    model.train()  # Configures the model for training

    for epoch in range(start_epoch, num_of_epochs):
        time.sleep(0.1)
        print('epoch [{}/{}] started.'.format(epoch + 1, num_of_epochs))
        time.sleep(0.1)
        total_train_loss, total_train_correct, total_train_no = 0, 0, 0
        for train_no, (train_inputs, train_labels) in enumerate(tqdm(train_loader)):
            img = train_inputs.to(device, non_blocking=True)
            labels = train_labels.to(device, non_blocking=True)
            decoded, class_out = model(img)
            loss_decoder = criterion_decoder(decoded, img)
            loss_class_out = criterion_classifier(class_out, labels)
            loss = loss_decoder + loss_ratio * loss_class_out
            _, predicted = torch.max(class_out.data, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss) * img.size()[0]
            total_train_correct += (predicted == labels).sum().item()
            total_train_no += img.size()[0]
            if train_no % (len(train_loader) // tests_per_epoch) == (len(train_loader) // tests_per_epoch) - 1:
                total_test_loss, total_test_correct, total_test_no = 0, 0, 0
                test_image = None
                model.eval()  # Configures the model for testing
                for test_no, (test_inputs, test_labels) in enumerate(test_loader):
                    with torch.no_grad():
                        img = test_inputs.to(device, non_blocking=True)
                        labels = test_labels.to(device, non_blocking=True)
                        decoded, class_out = model(img)
                        loss_decoder = criterion_decoder(decoded, img)
                        loss_class_out = criterion_classifier(class_out, labels)
                        loss = loss_decoder + loss_ratio * loss_class_out
                        _, predicted = torch.max(class_out.data, 1)
                    total_test_loss += float(loss) * img.size()[0]
                    total_test_correct += (predicted == labels).sum().item()
                    total_test_no += img.size()[0]
                    if test_no == 0:
                        test_image = torch.cat((img.data[:tests_per_row], decoded.data[:tests_per_row]), dim=0)
                        for i in range(1, tests_per_image // tests_per_row):
                            test_image = torch.cat((test_image,
                                                    img.data[i*tests_per_row:(i+1)*tests_per_row],
                                                    decoded.data[i*tests_per_row:(i+1)*tests_per_row]), dim=0)
                train_losslogger.append(total_train_loss / total_train_no)
                test_losslogger.append(total_test_loss / total_test_no)
                train_acclogger.append(total_train_correct / total_train_no)
                test_acclogger.append(total_test_correct / total_test_no)
                test_number = str(train_no // (len(train_loader) // tests_per_epoch) + 1)\
                    .zfill(int(math.log10(tests_per_epoch)) + 1)
                time_logger.append(str(epoch + 1) + '_' + test_number)
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'time_logger': time_logger,
                         'train_losslogger': train_losslogger, 'test_losslogger': test_losslogger,
                         'train_acclogger': train_acclogger, 'test_acclogger': test_acclogger, }
                if not os.path.isdir('./dc_output'):
                    os.mkdir('./dc_output')
                torch.save(state, './dc_output/conv_autoencoder_{}_{}.pth'
                           .format(epoch + 1, test_number))
                save_image(test_image, './dc_output/test_{}_{}_'
                                       'train_loss_{:.4f}_test_loss_{:.4f}_train_acc_{:.2f}_test_acc_{:.2f}.jpg'
                           .format(epoch + 1, test_number,
                                   total_train_loss / total_train_no, total_test_loss / total_test_no,
                                   total_train_correct / total_train_no, total_test_correct / total_test_no),
                           nrow=tests_per_row, normalize=True)
                model.train()  # Configures the model for training
                # total_train_loss, total_train_correct, total_train_no = 0, 0, 0  # zeroes out totals after every test

    print("Training completed with {} epochs.".format(num_of_epochs))


def plot(filename=None):
    assert os.path.isdir('./dc_output'), 'The folder "dc_output" is not found.'
    if filename is None:
        files = [file for file in os.listdir('./dc_output') if file.endswith(".pth")]
        assert len(files) > 0, 'There is no checkpoint to load from.'
        max_len = len(max(files, key=len))
        filename = max([file for file in files if len(file) == max_len])
    checkpoint = torch.load(os.path.join('./dc_output', filename))
    time_logger = checkpoint['time_logger']
    train_losslogger = checkpoint['train_losslogger']
    test_losslogger = checkpoint['test_losslogger']
    train_acclogger = checkpoint['train_acclogger']
    test_acclogger = checkpoint['test_acclogger']

    plt.close(fig='all')
    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    plt.xticks(np.arange(len(time_logger)), time_logger, rotation=90)

    tick_no = 10  # number of ticks on y axis

    min_y = min(min(train_losslogger, test_losslogger))
    max_y = max(max(train_losslogger, test_losslogger))
    base = round((max_y - min_y) / tick_no, 1 - math.floor(math.log10(max_y - min_y)))
    ax1.plot(np.arange(len(time_logger)), train_losslogger, 'g.-', label='Train loss')
    ax1.plot(np.arange(len(time_logger)), test_losslogger, 'r.-', label='Test loss')
    ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=base))
    ax1.margins(x=0.01, y=0.05)
    ax1.legend(loc='upper right')
    ax1.grid(linestyle='--', linewidth='0.5', alpha=0.7)

    min_y = min(min(train_acclogger, test_acclogger))
    max_y = max(max(train_acclogger, test_acclogger))
    base = round((max_y - min_y) / tick_no, 1 - math.floor(math.log10(max_y - min_y)))
    ax2.plot(np.arange(len(time_logger)), train_acclogger, 'g.-', label='Train accuracy')
    ax2.plot(np.arange(len(time_logger)), test_acclogger, 'r.-', label='Test accuracy')
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=base))
    ax2.margins(x=0.01, y=0.05)
    ax2.legend(loc='lower right')
    ax2.grid(linestyle='--', linewidth='0.5', alpha=0.7)

    def on_resize(event):
        fig.tight_layout()
        fig.canvas.draw()

    fig.canvas.mpl_connect('resize_event', on_resize)
    fig.set_size_inches(17, 8)
    plt.gcf().canvas.set_window_title('Logger')
    plt.get_current_fig_manager().window.showMaximized()
    plt.savefig('Logger.png', bbox_inches='tight', dpi=160)
    print("=> plotted checkpoint '{}'".format(filename))
    plt.show()


if __name__ == '__main__':
    train(continue_from_last=True, num_of_epochs=15, train_on_gpu=True)
    plot()
    pass
