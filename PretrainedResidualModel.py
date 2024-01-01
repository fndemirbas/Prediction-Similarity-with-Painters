import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_CLASSES = 50


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def confusion_matrix(outputs, labels):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    _, preds = torch.max(outputs, dim=1)

    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1

    return cm


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def test_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        cm = confusion_matrix(out, labels)
        return {"test_loss": loss.detach(), "test_acc": acc, "confusion_matrix": cm}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def test_epoch_end(self, outputs):
        batch_losses = [x["test_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["test_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        batch_cms = [x["confusion_matrix"] for x in outputs]
        epoch_cm = torch.stack(batch_cms).sum(dim=0).numpy().tolist()
        return {
            "test_loss": epoch_loss.item(),
            "test_acc": epoch_acc.item(),
            "confusion_matrix": epoch_cm,
        }

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class PretrainedResidualModel(ImageClassificationBase):
    def __init__(self, num_classes, with_last_cnn=True):
        super(PretrainedResidualModel, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # freezing the pretrained layers
        freeze_till = 7 if with_last_cnn else 8
        for idx, child in enumerate(self.model.children()):
            if idx < freeze_till:
                for params in child.parameters():
                    params.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
