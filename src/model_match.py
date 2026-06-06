import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("models", exist_ok=True)

class MatchClassifier(nn.Module):
    def __init__(self, input_size):
        super(MatchClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.sigmoid(self.fc4(out))
        return out

def train_match_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    model = MatchClassifier(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("\nTraining Match Outcome model...")
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_test)
            val_loss = criterion(val_output, y_test)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/match_outcome.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = (model(X_test) > 0.5).float()
        accuracy = (y_pred == y_test).float().mean()
        print(f"\n  Match Accuracy: {accuracy.item()*100:.2f}%")

    y_pred_np = y_pred.numpy().astype(int).flatten()
    y_test_np = y_test.numpy().astype(int).flatten()
    print("\n  Classification Report:")
    print(classification_report(y_test_np, y_pred_np,
          target_names=["Team B Wins", "Team A Wins"]))

    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_test_np, y_pred_np)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Team B Wins", "Team A Wins"],
                yticklabels=["Team B Wins", "Team A Wins"])
    plt.title("Match Outcome Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("plots/match_confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Match Outcome Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/match_loss.png")
    plt.close()

    return model