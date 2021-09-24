
# Data augmentation and normalization for training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import tqdm
from PIL import Image
import pandas as pd


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]


# Just normalization for validation
def get_train_transform():
    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return(trans)

def get_val_transform():
    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return(trans)



def get_model(num_class):
    model = models.vgg11(pretrained=True)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        #nn.Linear(num_ftrs, 500),
        nn.Linear(num_ftrs, num_class)
    )
    return(model)



def get_data_loader(data, batch_size, num_workers, train=True):
    if train:
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=train)
    else:
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=train)
        
    return(data_loader)



def train_model(model, epochs, dataloader, device, criterion, optimizer, scheduler, model_save_path):

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        loss_list = []
        acc_list = []
        itr = 1
        p_itr = 200

        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            
            if itr%p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0
                
            itr += 1

    #plt.plot(loss_list, label='loss')
    #plt.plot(acc_list, label='accuracy')
    #plt.legend()
    #plt.title('training loss and accuracy')
    #plt.show()

    torch.save(model.state_dict(), model_save_path)
    print("model saved")

    return([total_loss, loss_list, acc_list])


def accuracy_score(actual, pred):
    collate = 0
    dog = 0
    cat = 0

    pred = pred.cpu()

    for i in range(len(actual)):
        if actual[i] == pred[i]:
            collate+=1
        if actual[i] == pred[i] == 0:
            dog += 1
        if actual[i] == pred[i] == 1:
            cat += 1

    collate = collate / len(actual) * 100.0
    dog = dog / np.sum(actual == 0) * 100.0
    cat = cat / np.sum(actual == 1) * 100.0

    return([collate, dog, cat])



def eval_model(model, test_loader, device):
    model.eval()

    pred_list = []

    acc_list = []
    dog_list = []
    cat_list = []

    for x, fn in test_loader:
        with torch.no_grad():
            x = x.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            pred_list += [p.item() for p in pred]
            scores = accuracy_score(fn, pred)

            acc_list.append(scores[0])
            dog_list.append(scores[1])
            cat_list.append(scores[2])

    submission = pd.DataFrame({"total": acc_list, "dog": dog_list, "cat": cat_list})
    submission.to_csv('metrics.csv', index=False)