import torch
from dataset import PlantDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
import time
import pandas as pd
from torchmetrics import Precision, Recall
import warnings
from torchvision.models import resnet18
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = 'warn'


def train_model(model, 
                train_dataset, 
                valid_dataet, 
                batch_size, 
                epochs, 
                learning_rate,
                model_name,
                weight_decay = 0.00005
                ):
    
    torch.cuda.empty_cache()


    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)


    testloader = DataLoader(valid_dataet, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    model = model.to('cuda:0')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    precisions = []
    precisions_val = []

    recalls = []
    val_recalls = []

    losses = []
    val_losses = []


    metric_precision = Precision(task="multiclass", num_classes=4, average=None).to('cuda')
    metric_recall = Recall(task="multiclass", num_classes=4, average=None).to('cuda')

    metric_precision_val = Precision(task="multiclass", num_classes=4, average=None).to('cuda')
    metric_recall_val = Recall(task="multiclass", num_classes=4, average=None).to('cuda')

    for epoch in range(epochs):
        recall = []
        precision = []

        recall_val = []
        precision_val = []

        training_loss = []
        start_time = time.time()
        elapsed_time = 0
        model.train() 
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()

            inputs, labels = data 
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, outputs = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)

            metric_precision(outputs, labels)
            metric_recall(outputs, labels)

            loss.backward()
            optimizer.step()
            
            training_loss.append(loss.item())

            if (i + 1) % 5 == 0 or i == len(trainloader) - 1:
                elapsed_time = time.time() - start_time
                batches_done = i + 1
                batches_total = len(trainloader)
                batches_remaining = batches_total - batches_done
                time_per_batch = elapsed_time / batches_done
                estimated_time_remaining = time_per_batch * batches_remaining

                elapsed_time_minutes = elapsed_time / 60
                estimated_time_remaining_minutes = estimated_time_remaining / 60

                progress_message = f'Batch {i}/{len(trainloader)},Remaining: {estimated_time_remaining_minutes:.2f}min'
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

            

    
        model.eval()  
        val_loss = []
        
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                labels = torch.Tensor(labels)
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')


                outputs = model(inputs)
                val_loss_crt = criterion(outputs, labels)

                _, outputs = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)

                metric_precision_val(outputs, labels)
                metric_recall_val(outputs, labels)


                val_loss.append(val_loss_crt.item())

        precision = metric_precision.compute()
        recall = metric_recall.compute()

        precision_val = metric_precision_val.compute()
        recall_val = metric_recall_val.compute()

        precisions.append(precision)
        precisions_val.append(precision_val)

        recalls.append(recall)
        val_recalls.append(recall_val)
        
        losses.append(np.mean(training_loss))
        val_losses.append(np.mean(val_loss))
        
        print(f'Epoch {epoch + 1}, Training loss: {np.mean(training_loss)} Validation Loss: {np.mean(val_loss)}')
        
        print(f'Epoch {epoch + 1}, Training Class 1: {precision[0]}, Class 2: {precision[1]}, Class 3: {precision[2]}, Class 4: {precision[3]}')
        print(f'Epoch {epoch + 1}, Validation Class 1: {precision_val[0]}, Class 2: {precision_val[1]}, Class 3: {precision_val[2]}, Class 4: {precision_val[3]}')

        print(f'Epoch {epoch + 1}, Training Class 1: {recall[0]}, Class 2: {recall[1]}, Class 3: {recall[2]}, Class 4: {recall[3]}')
        print(f'Epoch {epoch + 1}, Validation Class 1: {recall_val[0]}, Class 2: {recall_val[1]}, Class 3: {recall_val[2]}, Class 4: {recall_val[3]}')

        

    print('Finished Training')
    print('Finished Training') 

    torch.save(model.state_dict(),f'models/{model_name}.pth')
    df = pd.DataFrame()
    df['loss'] = np.array(losses)
    df['val_loss'] = np.array(val_losses)

    df['recall'] = [list(recall.cpu().numpy()) for recall in recalls]
    df['val_recall'] = [list(val_recall.cpu().numpy()) for val_recall in val_recalls]

    df['precision'] = [list(precision.cpu().numpy()) for precision in precisions] 
    df['val_precision'] = [list(val_precision.cpu().numpy()) for val_precision in precisions_val]
    df.to_csv(f'results/{model_name}.csv')
