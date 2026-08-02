import torch

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score, ConfusionMatrixDisplay

from tqdm import tqdm

from .utils import transform_embedding

def train_epoch(backbone, 
                classifier, 
                optimizer, 
                dataloader, 
                loss_fn,
                device,
            ):
    backbone.eval()
    classifier.train()

    total_loss = 0.
    preds_log = []
    labels_log = []

    for data in tqdm(dataloader):
        optimizer.zero_grad()

        seq = data['seq'].to(device)
        mask = data['mask'].to(device)
        label = data['label'].to(device).squeeze()

        with torch.no_grad():
            embeddings = backbone(seq)
            embeddings = transform_embedding(embeddings, mask)

        preds = classifier(embeddings)
        loss = loss_fn(preds, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()
        preds_log.extend(preds.argmax(1).detach().cpu().tolist())
        labels_log.extend(label.cpu().tolist())

    accuracy = accuracy_score(labels_log, preds_log)
    f1 = f1_score(labels_log, preds_log, average='macro')
    recall = recall_score(labels_log, preds_log, average='macro')
    precision = precision_score(labels_log, preds_log, average='macro')    

    disp = ConfusionMatrixDisplay.from_predictions(
        labels_log,
        preds_log,
        colorbar=False,
        normalize="true"  
    )

    return {
        'loss': total_loss,
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': disp
    }

def validation_epoch(backbone, 
                    classifier, 
                    dataloader, 
                    loss_fn,
                    device,
            ):
    backbone.eval()
    classifier.eval()

    total_loss = 0.
    preds_log = []
    labels_log = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            seq = data['seq'].to(device)
            mask = data['mask'].to(device)
            label = data['label'].to(device).squeeze()

            embeddings = backbone(seq)
            embeddings = transform_embedding(embeddings, mask)

            preds = classifier(embeddings)
            loss = loss_fn(preds, label)

            total_loss += loss.detach().cpu().numpy().item()
            preds_log.extend(preds.argmax(1).detach().cpu().tolist())
            labels_log.extend(label.cpu().tolist())

    accuracy = accuracy_score(labels_log, preds_log)
    f1 = f1_score(labels_log, preds_log, average='macro')
    recall = recall_score(labels_log, preds_log, average='macro')
    precision = precision_score(labels_log, preds_log, average='macro')    

    disp = ConfusionMatrixDisplay.from_predictions(
        labels_log,
        preds_log,
        colorbar=False,
        normalize="true"  
    )

    return {
        'loss': total_loss,
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': disp
    }

def log_to_tensorboard(writer, step, log, dataset, label):
    writer.add_scalar(f'Classifier/{dataset}/Loss/{label}', log['loss'], step)
    writer.add_scalar(f'Classifier/{dataset}/Accuracy/{label}', log['accuracy'], step)
    writer.add_scalar(f'Classifier/{dataset}/F1/{label}', log['f1'], step)
    writer.add_scalar(f'Classifier/{dataset}/Recall/{label}', log['recall'], step)
    writer.add_scalar(f'Classifier/{dataset}/Precision/{label}', log['precision'], step)

    writer.add_figure(
        f'Classifier/{dataset}/Confusion Matrix/{label}',
        log['confusion_matrix'].figure_,
        global_step=step,
    )

    plt.close(log['confusion_matrix'].figure_)

def train_model(backbone, 
                classifier, 
                optimizer, 
                train_dataloader, 
                val_dataloader,
                loss_fn,
                log_writer,
                epochs,
                device,
                dataset_name
            ):
    for epoch in range(1, epochs + 1):
        train_log = train_epoch(
                backbone=backbone, 
                classifier=classifier, 
                optimizer=optimizer, 
                dataloader=train_dataloader, 
                loss_fn=loss_fn,
                device=device) 
        log_to_tensorboard(log_writer, epoch, train_log, dataset_name,'train')

        val_log = validation_epoch(
            backbone=backbone,
            classifier=classifier,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device)
        log_to_tensorboard(log_writer, epoch, val_log, dataset_name, 'val')
    
    return val_log