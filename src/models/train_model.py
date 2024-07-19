from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model, train_loader, loss_fn, optimizer, print_freq=20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm()}")

        running_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % print_freq == 0 or i == len(train_loader):
            avg_loss = running_loss / i
            accuracy = 100. * correct / total
            print(
                f"Batch {i}/{len(train_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    running_loss = running_loss / len(train_loader)
    accuracy_final = 100. * correct / total
    return running_loss, accuracy_final


def test_step(model, test_loader, loss_fn, print_freq=20):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for i, (inputs, labels) in enumerate(test_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % print_freq == 0 or i == len(test_loader):
                avg_loss = running_loss / i
                accuracy = 100. * correct / total
                print(
                    f"Batch {i}/{len(test_loader)}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    running_loss = running_loss / len(test_loader)
    accuracy_final = 100. * correct / total
    return running_loss, accuracy_final


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn,
          optimizer: torch.optim,
          save_name,
          epochs: int = 20,
          print_freq: int = 1):

    def compute_confusion_matrix(model, data_loader, device):
        model.to(device)
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                # predicted = torch.round(torch.sigmoid(outputs))
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        print(classification_report(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        return cm, y_true, y_pred

    # Define your learning rate scheduler
    loss_list = []
    acc_list = []
    loss_train_list = []
    acc_train_list = []
    best_performance = float('-inf')
    best_epoch = None
    model.to(device)
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model, train_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, print_freq=print_freq)
        test_loss, test_acc = test_step(
            model=model, loss_fn=loss_fn, test_loader=test_dataloader, print_freq=print_freq)

        loss_list.append(test_loss)
        acc_list.append(test_acc)
        loss_train_list.append(train_loss)
        acc_train_list.append(train_acc)

        if test_acc > best_performance:
            best_performance = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'{save_name}.pth')

    # 4. Print out what's happening
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
    )

    # 5. Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    plt.plot(range(1, epochs+1), acc_train_list, label="Acc_Train")
    plt.plot(range(1, epochs+1), acc_list, label="Acc_Val")
    plt.legend()
    plt.show()
    plt.plot(range(1, epochs+1), loss_list, label="loss_Val")
    plt.plot(range(1, epochs+1), loss_train_list, label="loss_Train")
    plt.legend()
    plt.show()
    cm, y_true, y_pred = compute_confusion_matrix(
        model, test_dataloader, device)
    print(cm)

    # 6. Return the filled results at the end of the epochs
    print(
        f"Best epoch: {best_epoch+1}, Best test accuracy: {best_performance:.2f}%")
    return results
