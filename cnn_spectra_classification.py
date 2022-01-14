import torch
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from astropy.io import fits
import numpy as np
import glob
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from glob import glob
import warnings



def read_lamost(paths, flux, scls):
    paths = glob(paths + '*.fits')
    wav = 3700
    for idx, file in enumerate(paths):
        with fits.open(file) as hdulist:
            f = hdulist[0].data[0]
            f = f[:wav]
            f = (f - np.min(f)) / (np.max(f) - np.min(f))
            f = np.array([np.array([f])])
            s = hdulist[0].header['SUBCLASS'][0] # Change manually in advance.
        flux.append(f)
        scls.append(s)
    return flux, scls

def read_data(file_all):
    ''' Reads in the flux and classes from LAMOST fits files & converts classes to one-hot vectors '''
    print("Reading in LAMOST data...")
    flux = []
    scls = []
    flux, scls = read_lamost(file_all, flux, scls)
    flux = np.array(flux)
    cls = onehot(scls)
    scls = np.array(cls)
    fluxTR, fluxTE, clsTR, clsTE = train_test_split(flux, scls, test_size=0.3)
    Xtrain1 = torch.from_numpy(fluxTR)
    Xtest1 = torch.from_numpy(fluxTE)
    ytrain1 = torch.from_numpy(clsTR)
    ytest1 = torch.from_numpy(clsTE)
    # Save data sets here for binary classification models.
    torch_dataset_train = Data.TensorDataset(Xtrain1, ytrain1)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    data_loader_train = torch.utils.data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_test

def onehot(classes):
    ''' Encodes a list of descriptive labels as one hot vectors '''
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(classes)
    labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded) + 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    return onehot_encoded

def get_variable(x):
    x = Variable(x)
    return x

class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=(1, 4), stride=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 40, kernel_size=(1, 5), stride=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 50, kernel_size=(1, 7), stride=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 60, kernel_size=(1, 9), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.dense = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(60 * 1 * 224, 1024),
            torch.nn.Linear(1024, num_class),
            torch.nn.Softmax())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 60 * 1 * 224)
        x = self.dense(x)
        return x


def run_CNN_module(device, num_epochs, learning_rate, train, test):
    # Training and parameter optimization of the model.
    cnn_model = CNN_Model()
    cnn_model = cnn_model.to(device=device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(num_epochs):
        TR_loss = 0.0
        for data in train:
            optimizer.zero_grad()
            X_train, y_train = data
            X_train, y_train = get_variable(X_train), get_variable(y_train)
            optimizer.zero_grad()
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_train = X_train.to(device=device)
            y_train = y_train.to(device=device)
            outputs = cnn_model(X_train)
            _, pred = torch.max(outputs.data, 1)
            y_train = torch.max(y_train, 1)[1]
            lossTR = loss_func(outputs, y_train)
            lossTR.backward()
            optimizer.step()
            TR_loss += lossTR.item()
        for data in test:
            X_test, y_test = data
            X_test, y_test = get_variable(X_test), get_variable(y_test)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            X_test = X_test.to(device=device)
            y_test = y_test.to(device=device)
            outputs = cnn_model(X_test)
            _, pred = torch.max(outputs.data, 1)
            y_test = torch.max(y_test, 1)[1]

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = "cuda:0"
    num_class = 8
    num_epochs = 2500
    batch_size = 100
    learning_rate = 0.0001
    file = '/.../'
    train, test = read_data(file)
    run_CNN_module(device, num_epochs, learning_rate, train, test)
