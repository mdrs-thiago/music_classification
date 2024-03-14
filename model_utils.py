from tqdm import tqdm 
from torch.autograd import Variable
import torch 
import torch.nn as nn
import os 
import torchvision.transforms as transforms
import numpy as np 

def train_from_hf(model, train_loader, val_loader, optimizer, criterion, num_epochs, name='', device='cpu', **kwargs):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        min_val = 10000
        model.to(device);
        # Training loop
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=num_train_batches, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            print(inputs.shape)
            print(labels.shape)
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_acc += (predicted == labels).sum().item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(val_loader), total=num_val_batches, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).type(torch.LongTensor).to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_acc += (predicted == labels).sum().item()

        if val_loss < min_val:
            torch.save(model.state_dict(), os.path.join('ckpt',f'{name}_best_model.pth'))

        train_loss /= num_train_batches
        train_acc /= len(train_loader.dataset)
        val_loss /= num_val_batches
        val_acc /= len(val_loader.dataset)

        tqdm.write(f'Epoch {epoch + 1}/{num_epochs} - Training accuracy: {train_acc:.4f} - Training loss: {train_loss:.4f} - Validation accuracy: {val_acc:.4f} - Validation loss: {val_loss:.4f}')
