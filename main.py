from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import os


'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 12, 3, 1)
        self.avgpool = nn.AvgPool2d(kernel_size = (2,2), stride =1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1452, 200)
        self.fc2 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)


        # print(np.shape(x))
        # print("conv1")

        
        x = self.conv2(x)
        x = F.relu(x)


        # print(np.shape(x))
        # print("conv2")
        
        x = self.avgpool(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # print(np.shape(x))
        # print("avgpool")
        
        
        x = torch.flatten(x, 1)
        # print(np.shape(x))
        # print("flatte")
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        
        output = F.log_softmax(x, dim=1)
        return output
    
    def to_feature(self, x):
        x = self.conv1(x)
        x = F.relu(x)


        # print(np.shape(x))
        # print("conv1")

        
        x = self.conv2(x)
        x = F.relu(x)


        # print(np.shape(x))
        # print("conv2")
        
        x = self.avgpool(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # print(np.shape(x))
        # print("avgpool")
        
        
        x = torch.flatten(x, 1)
        # print(np.shape(x))
        # print("flatte")
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

def find_errors(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    sk_confusion = np.zeros((10,10))
    confusion = np.zeros((10,10))
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            printed = 0
            # if printed == 0:
            #     print(pred.eq(target.view_as(pred)))
            #     printed = 1
            # print(data)
            # img = Image.fromarray(np.array(data))
            # img.show()
            # image
            # print(target)
            # index = 0
            # for image in pred.eq(target.view_as(pred)):
            #     if not image:
            #         plt.imshow(np.array(torch.Tensor.cpu(data[index,0:,:])).reshape((28,28)))
            #         plt.show()
                    
            #     index +=1
            i= 0
            sk_confusion += confusion_matrix(torch.Tensor.cpu(target), torch.Tensor.cpu(pred))
            for guess in pred:
                confusion[target[i],guess] += 1
                i += 1
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
    print(confusion)        
    print(sk_confusion)
    plt.imshow(np.log(sk_confusion))
    plt.show()
    
    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))



def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        print(args)
        print(args.load_model)
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('./data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        find_errors(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.RandomRotation(5, fill=(0,)),
                    transforms.RandomPerspective(.15, p = .15),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                    
                    
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    sorted_train = [[],[],[],[],[],[],[],[],[],[]]
    index = 0
    portion = 1
    
    random_portion = torch.utils.data.RandomSampler(train_dataset)
    for image in train_dataset:
        sorted_train[image[1]].append(index)
        index += 1

    subset_indices_train = []
    subset_indices_valid = []
    for number in sorted_train:
        number_randomized = list(torch.utils.data.SubsetRandomSampler(number))[:int(len(number)*(portion))]
        subset_indices_valid += number_randomized[:int(len(number_randomized)*.15)]
        subset_indices_train += number_randomized[int(len(number_randomized)*.15):]


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler
    print("train set")
    test(model, device, train_loader)
    print("validation Set")
    test(model, device, val_loader)
    print(device)
        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
train_accuraccy =  1 -np.array([25261/25503,12579/12750 ,6273/6376 ,3126/3189])
test_accuraccy = 1-np.array([9875/10000, 9821/10000,9715/10000 ,9584/10000])
train_points = [25503,12750 ,6376 ,3189]
plt.plot(train_points,train_accuraccy)
plt.plot(train_points,test_accuraccy)
plt.legend(['Train Accuracy','Test Accuracy'])
plt.xscale("log")
plt.yscale('log')

device = torch.device("cuda")
model = Net().to(device)
model.to_feature()
#model.load_state_dict(torch.load('mnist_model.pt'))

# first_kernals = model.state_dict()['conv1.weight']
# for channel in first_kernals:
#     plt.figure()
#     plt.imshow(np.array(torch.Tensor.cpu(channel)).reshape((3,3)))
# plt.show()
