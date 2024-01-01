import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import cv2
from PIL import Image

from PretrainedResidualModel import PretrainedResidualModel


TEST_SIZE = 0.1
VAL_SIZE = 0.2
NUM_CLASSES = 49
IMG_SIZE = 64

CATEGORIES = [
    'Alfred Sisley',
    'Amedeo Modigliani',
    'Andrei Rublev',
    'Andy Warhol',
    'Camille Pissarro',
    'Caravaggio',
    'Claude Monet',
    'Diego Rivera',
    'Diego Velazquez',
    'Edgar Degas',
    'Edouard Manet',
    'Edvard Munch',
    'El Greco',
    'Eugene Delacroix',
    'Francisco Goya',
    'Frida Kahlo',
    'Georges Seurat',
    'Giotto di Bondone',
    'Gustav Klimt',
    'Gustave Courbet',
    'Henri de Toulouse-Lautrec',
    'Henri Rousseau',
    'Henri Matisse',
    'Hieronymus Bosch',
    'Jackson Pollock'
    'Jan van Eyck',
    'Joan Miro',
    'Kazimir Malevich',
    'Leonardo da Vinci',
    'Marc Chagall',
    'Michelangelo',
    'Mikhail Vrubel',
    'Pablo Picasso',
    'Paul Cezanne',
    'Paul Gauguin',
    'Paul Klee',
    'Peter Paul Rubens',
    'Pierre-Auguste Renoir',
    'Piet Mondrian',
    'Pieter Bruegel',
    'Raphael',
    'Rembrandt',
    'Rene Magritte',
    'Salvador Dali',
    'Sandro Botticelli',
    'Titian',
    'Vincent van Gogh',
    'William Turner',
    'Vasiliy Kandinskiy']


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_datasets(batch_size, dataset_dir):
    transform1 = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform2 = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform3 = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

    dataset1 = ImageFolder(dataset_dir, transform=transform1)
    dataset2 = ImageFolder(dataset_dir, transform=transform2)
    dataset3 = ImageFolder(dataset_dir, transform=transform3)
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

    dataset_size = len(dataset)
    print(dataset_size)

    val_size = int(dataset_size * VAL_SIZE)
    test_size = int(dataset_size * TEST_SIZE)
    train_size = dataset_size - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size * 2, num_workers=4, pin_memory=True)

    return train_dl, val_dl, test_dl


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in test_loader]
    return model.test_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(epochs, max_lr, model, train_loader, val_loader, test_loader, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []

    # Set up optimizer
    optimizer = opt_func(model.parameters(), max_lr)

    # Set up one-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    test_result = test(model, test_loader)

    results = {"training": history, "testing": test_result}

    return results


def plot_accuracies(history, save_path=None):
    plt.clf()
    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")

    if save_path:
        plt.savefig(save_path)


def plot_losses(history, save_path=None):
    plt.clf()
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")

    if save_path:
        plt.savefig(save_path)


def prediction(model_path, test_img_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.imread(test_img_path)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    im_pil = Image.fromarray(img)
    img = transform(im_pil)
    batch_t = torch.unsqueeze(img, 0)
    model = torch.load(model_path)
    model.eval()
    output = model(batch_t)  # Forward pass
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    print('Image predicted as ', CATEGORIES[pred.item()])


if __name__ == "__main__":
    test_path = 'test/image.jpg'
    model_path = 'model/model.pth'
    useSavedModel = 1   # If you want to use saved model, useSavedModel = 1, for start training again useSavedModel = 0.

    if useSavedModel == 1:
        prediction(model_path, test_path)
    else:
        epochs = 30
        lr = 0.001
        batch_size = 32
        dataset_dir = 'painter/images/images'

        train_dl, val_dl, test_dl = load_datasets(batch_size, dataset_dir)
        pretrainedResidualModel = PretrainedResidualModel(NUM_CLASSES).to(get_default_device())
        pretrainedResidualResults = train(epochs, lr, pretrainedResidualModel, train_dl, val_dl, test_dl)
        print(pretrainedResidualResults)

        pretrainedResidualHistory = pretrainedResidualResults["training"]
        plot_accuracies(pretrainedResidualHistory, 'graph/pretrainedResidualModel_acc.png')
        plot_losses(pretrainedResidualHistory, 'graph/pretrainedResidualModel_loss.png')
        pretrainedResidualTest = pretrainedResidualResults["testing"]
        print("Test Loss:", pretrainedResidualTest["test_loss"])
        print("Test Accuracy:", pretrainedResidualTest["test_acc"])

        df_cm = pd.DataFrame(pretrainedResidualTest["confusion_matrix"], range(50), range(50))
        sn.set(font_scale=0.8)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})  # font size
        plt.figure(figsize=(30, 30))
        plt.show()
