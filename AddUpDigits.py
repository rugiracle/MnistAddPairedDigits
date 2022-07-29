from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import numpy as np
from Helpers import dataGeneration as DG

N_CLASSES = 19
RANDOM_SEED = 50


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class SuNet(nn.Module):
    def __init__(self, n_classes):
        super(SuNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=60, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1080, out_features=420),
            nn.ReLU(),
            nn.Linear(in_features=420, out_features=n_classes),
        )

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.feature_extractor(x2)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def training_step(self, batch, samples_per_class):
        images_1, images_2, labels = batch
        out = self(images_1, images_2)  # Generate predictions
        """ Calculate weighted loss as dataset is imbalanced"""
        loss = F.cross_entropy(out, labels, samples_per_class, reduction='mean')
        return loss

    def validation_step(self, batch):
        images_1, images_2, labels = batch
        out = self(images_1, images_2)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.2f}, val_acc: {:.2f}".format(epoch, result['val_loss'], result['val_acc']))


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, samples_per_class, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    best_result = 0.0
    best_model = model
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch, samples_per_class)
            loss.backward()         # compute gradients
            optimizer.step()        # update weights
            optimizer.zero_grad()   # set grads to zero
        # Validation phase
        result = evaluate(model, val_loader)
        if result['val_acc'] > best_result:
            best_result = result['val_acc']
            best_model = model
        model.epoch_end(epoch, result)
        history.append(result)
    name = 'model/add_mnist_' + str(epochs) + '_' + str(lr)+'_' + str(int(best_result*100)) + '_best' + '.pt'
    torch.save(best_model.state_dict(), name)
    return history


def predict_image(img1, img2, device, model):
    xb1 = to_device(img1.unsqueeze(0), device)
    xb2 = to_device(img2.unsqueeze(0), device)
    yb = model(xb1, xb2)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


def training_model(epochs, lr, batch_size, device):
    """load data and preprocess them
        train the model,
        then save the model weights"""
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    labels_1, dataset1 = DG.pre_process_dataset_img_channel_Random(mnist_data)  # generate new training  dataset
    validation_prctg = 0.2  # split the training dataset into training and validation sets
    val_size = int(len(dataset1) * validation_prctg)
    train_size = len(dataset1) - val_size
    train_ds, val_ds = random_split(dataset1, [train_size, val_size])
    print('size of training set: {}, size of validation set: {}'.format(train_size, val_size))
    """compute the weights of samples in each batch """

    samples_per_class = list(labels_1.values())
    samples_per_class = DG.get_weights_inverse_num_of_samples(N_CLASSES, samples_per_class)
    samples_per_class = torch.tensor(samples_per_class).float()

    samples_per_class = samples_per_class.to(device)
    print(samples_per_class.device)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size)

    train_loader = DeviceDataLoader(train_loader, device)  # move data to device
    val_loader = DeviceDataLoader(val_loader, device)

    model = SuNet(n_classes=N_CLASSES).to(device)
    print('*************  training the model   *************')
    # Load the model
    #PATH = 'addition/mnist_sum_300.pt'
    #model = SuNet(n_classes=N_CLASSES).to(device)
    #model.load_state_dict(torch.load(PATH))
    #print('*************  Continuing training the model @' + PATH + '  *************')

    history = [evaluate(model, val_loader)]
    model.train()
    history += fit(epochs=epochs, lr=lr, model=model, train_loader=train_loader, val_loader=val_loader,
                   samples_per_class=samples_per_class)
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    name = 'model/loss_vs_epoch_' + str(epochs) + '_' + str(lr) + '_' + str(batch_size) + '.png'
    plt.savefig(name)
    plt.show()
    plt.clf()

    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    name = 'model/acc_vs_no_epoch_' + str(epochs) + '_' + str(lr) + '_' + str(batch_size) + '.png'
    plt.savefig(name)
    plt.show()
    plt.clf()
    name = 'model/add_mnist_' + str(epochs) + '_' + str(lr) + '_' + str(batch_size) + '.pt'
    torch.save(model.state_dict(), name)


def testing_model(batch_size, device, PATH):
    """load test data and preprocess them
     load the model, and evaluate the test dataset
     then save some test results for later display"""
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    mnist_test = datasets.MNIST('../data', train=False,
                                transform=transform)

    labels_2, dataset2 = DG.pre_process_dataset_img_channel_Random(mnist_test)  # generate new test  dataset
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size)
    test_loader = DeviceDataLoader(test_loader, device)

    #####
    # Load the model
    model = SuNet(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    print('*************  testing the model @' + PATH + '  *************')
    result = evaluate(model, test_loader)
    print(result)
    #   display and save some results
    ROW_IMG = 10
    N_ROWS = 5
    choices = random.sample(range(len(mnist_test)), ROW_IMG * N_ROWS + 1)
    fig = plt.figure()
    for i in range(1, ROW_IMG * N_ROWS + 1):
        test_img1, label1 = mnist_test[choices[i - 1]]
        test_img2, label2 = mnist_test[choices[i]]
        test_img = DG.pre_process_imgs(test_img1, test_img2)   # concatenate images only for plotting
        plt.subplot(N_ROWS, ROW_IMG, i)
        plt.axis('off')
        plt.imshow(test_img.permute((1, 2, 0)), cmap='gray_r')
        title = f'{predict_image(test_img1, test_img2, device, model)} ({str(label1 + label2)}) '
        plt.title(title, fontsize=7, color='b')
    fig.suptitle('AddMnist- predictions')
    plt.savefig('results/AddMnist_predictions.png')
    plt.show()
    compute_metrics(model, test_loader, device)


def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


def compute_metrics(model, test_loader, device):
    actuals, predictions = test_label_predictions(model, device, test_loader)
    print('Accuracy score: %f' % accuracy_score(actuals, predictions))

    confusion_matrix_ = confusion_matrix(actuals, predictions)
    disp_confusion = ConfusionMatrixDisplay(confusion_matrix_)
    disp_confusion.plot()
    plt.savefig('results/AddMnist_confusion_matrix.png')
    plt.show()
    confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]
    acc_per_class = confusion_matrix_.diagonal()*100
    plt.bar(range(len(acc_per_class)), acc_per_class, tick_label=range(len(acc_per_class)))
    plt.savefig('results/AddMnist_PerClassAccuracy.png')
    plt.show()
    print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))


def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # training_model(epochs=150, lr=0.55, batch_size=128, device=device)
    PATH = 'model/add_mnist_150_0.55_128.pt'
    # PATH = 'model/add_mnist_150_0.55_95_best.pt'
    testing_model(batch_size=1000, device=device, PATH=PATH)


if __name__ == '__main__':
    main()