import torch
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

def train(num_epoch, model, train_loader, val_loader, criterion,
          optimizer, scheduler, save_dir, device):
    print("Strart training.........")
    running_loss = 0.0
    total = 0
    best_loss = 9999
    for epoch in range(num_epoch+1) :
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device) , labels.to(device)

            output = model(img)
            loss = criterion(output, label)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, argmax = torch.max(output, 1)
            acc = (label==argmax).float().mean()
            total += label.size(0)

            if (i + 1 ) % 10 == 0:
                print("Epoch [{}/{}] Step[{}/{}] Loss :{:.4f} Acc : {:.2f}%".format(
                    epoch + 1 , num_epoch, i+1, len(train_loader), loss.item(),
                    acc.item() * 100
                ))


        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion,
                                        device)
        # if epoch % 10 == 0:
        #     save_model(model, save_dir, file_naem=f"{epoch}.pt")
        if avrg_loss < best_loss :
            print(f"Best save at epoch >> {epoch}")
            print("save model in " , save_dir)
            best_loss = avrg_loss
            save_model(model, save_dir)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")

    save_model(model, save_dir, file_name="last_resnet.pt")
