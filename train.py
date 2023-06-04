import torch
import torch.nn as nn
from torch.optim import Adam
from utils import train_model,evaluate_model
from model import GenderClassifier
import config
import matplotlib.pyplot as plt

from data_loader import vocab_size,train_loader,validation_loader,char_to_index
device = config.DEVICE
print(f"Using {device} Device")

model = GenderClassifier(vocab_size)
trainable_params = sum([torch.numel(i) for i in model.parameters() if i.requires_grad])
print(f"Total number of trainable parameters: {trainable_params}\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr = config.LEARNING_RATE,
    # weight_decay=0.01
    )


model = model.to(device)

num_epochs = config.NUM_EPOCHS
checkpoint_path = config.CHECKPOINT_FILE

best_val_accuracy = 0.0
patience = 0

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_epoch_loss, train_epoch_accuracy = train_model(model, train_loader, loss_fn, optimizer, device)
    val_epoch_loss, val_epoch_accuracy = evaluate_model(model, validation_loader, loss_fn, device)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

    print(f"TRAIN: \t Epoch {epoch+1}/{num_epochs}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.2f}%")
    print(f"VALIDATION: \t Epoch {epoch+1}/{num_epochs}, Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")

    if val_epoch_accuracy > best_val_accuracy:
        patience = 0
        best_val_accuracy = val_epoch_accuracy
        model_dict = {
            "state_dict": model.state_dict(),
            "char_to_index": char_to_index,
            "params": {k: v for k, v in config.__dict__.items() if "__" not in k}
        }
        torch.save(model_dict, checkpoint_path)
        print("Checkpoint saved.")
    else:
        patience += 1
        if patience >= config.PATIENCE:
            print("Early stopping. No improvement in validation accuracy.")
            break
        
print("Training finished.")

# Plotting the curves
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))

# Plotting loss curves
plt.subplot(1, 2, 1)
plt.plot(epochs[:len(train_losses)], train_losses, label="Train")
plt.plot(epochs[:len(val_losses)], val_losses, label="Validation")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plotting accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs[:len(train_accuracies)], train_accuracies, label="Train")
plt.plot(epochs[:len(val_accuracies)], val_accuracies, label="Validation")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Saving the figure
plt.savefig("results/training_curves.png")
plt.show()

