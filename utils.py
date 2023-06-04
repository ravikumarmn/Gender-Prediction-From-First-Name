import torch

from tqdm import tqdm


def train_model(model,train_loader,loss_fn,optimizer,device):
    running_loss = 0.0
    total_sample = 0
    correct_prediction = 0
    model.train()
    tqdm_train_loader = tqdm(train_loader,total = len(train_loader))
    for batch in tqdm_train_loader:
        input_ids = batch['fname'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()

        logits = model(input_ids)

        loss = loss_fn(logits,label)
        tqdm_train_loader.set_postfix({"LOSS": float(f"{loss.item():.3f}")})
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted_indices = torch.max(logits.data,1)
        total_sample += label.size(0)
        correct_prediction += (predicted_indices == label).sum().item()
    train_epoch_loss = running_loss / len(train_loader)
    train_epoch_accuracy = (correct_prediction / total_sample)* 100.0
    return train_epoch_loss,train_epoch_accuracy


def evaluate_model(model,validation_loader,loss_fn,device):
    running_loss = 0.0
    total_sample = 0
    correct_prediction = 0
    model.eval()
    tqdm_validation_loader = tqdm(validation_loader,total = len(validation_loader))
    with torch.no_grad():
        for batch in tqdm_validation_loader:
            input_ids = batch['fname'].to(device)
            label = batch['label'].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits,label)
            tqdm_validation_loader.set_postfix({"LOSS": float(f"{loss.item():.3f}")})

            running_loss += loss.item()
            _,predicted_indices = torch.max(logits.data,1)
            total_sample += label.size(0)
            correct_prediction += (predicted_indices == label).sum().item()
        val_epoch_loss = running_loss / len(validation_loader)
        val_epoch_accuracy = (correct_prediction / total_sample)* 100.0
        return val_epoch_loss,val_epoch_accuracy