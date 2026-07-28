import torch

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score, ConfusionMatrixDisplay

from tqdm import tqdm

def transform_embedding(embeddings, mask):
    embeddings = embeddings.permute(0, 2, 3, 1)
    mask = mask.unsqueeze(1).unsqueeze(1).float().to(embeddings.device)
    embeddings = (embeddings * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
    embeddings = embeddings.flatten(1)

    return embeddings

def train_epoch(dataloader, backbone, classifier, loss_fn, optimizer, device):
    backbone.train()
    classifier.train()

    total_loss = 0.
    preds_log = []
    labels_log = []

    for data in tqdm(dataloader):
        optimizer.zero_grad()

        seq = data['seq'].to(device)
        mask = data['mask'].to(device)
        label = data['label'].to(device)

        embeddings = backbone(seq)
        embeddings = transform_embedding(embeddings, mask)

        preds = classifier(embeddings)
        print('='*100)
        print(preds.shape, label.shape)
        print('='*100)
        loss = loss_fn(preds, label)

        loss.backwards()
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


def validation_epoch(dataloader, backbone, classifier, loss_fn, device):
    backbone.eval()
    classifier.eval()

    total_loss = 0.
    preds_log = []
    labels_log = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            seq = data['seq'].to(device)
            mask = data['mask'].to(device)
            label = data['label'].to(device)

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
    writer.add_scalar(f'{dataset}/Loss/{label}', log['loss'], step)
    writer.add_scalar(f'{dataset}/Accuracy/{label}', log['accuracy'], step)
    writer.add_scalar(f'{dataset}/F1/{label}', log['f1'], step)
    writer.add_scalar(f'{dataset}/Recall/{label}', log['recall'], step)
    writer.add_scalar(f'{dataset}/Rrecision/{label}', log['recall'], step)

    writer.add_figure(
        f'{dataset}/Confusion Matrix/{label}',
        log['confusion_matrix'].figure_,
        global_step=step,
    )

    plt.close(log['confusion_matrix'].figure_)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad_(True)

def train_model(train_dataloader, 
                val_dataloader, 
                backbone, 
                classifier, 
                loss_fn, 
                optimizer, 
                epochs, 
                log_writer,
                device,
                dataset_name,
                *, 
                train_backbone=True):

    if not train_backbone:
        freeze_model(backbone)
    else:
        unfreeze_model(backbone)

    for epoch in range(1, epochs + 1):
        train_log = train_epoch(train_dataloader, backbone, classifier, loss_fn, optimizer, device)
        log_to_tensorboard(log_writer, epoch, train_log, dataset_name,'train')

        val_log = validation_epoch(val_dataloader, backbone, classifier, loss_fn, device)
        log_to_tensorboard(log_writer, epoch, val_log, dataset_name, 'val')

    