import torch
import torch.nn as nn
from data.dataloader import create_dataloader
from models.detector import create_model
import pickle
from tqdm import tqdm


def train_one_epoch(model, device, train_dataloader, criterion, optimizer, epoch):

    model.train()
    
    running_loss = 0

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    average_loss = running_loss/ len(train_dataloader)
    return average_loss


def validate(model, device, validation_dataloader, criterion, epoch):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(validation_dataloader, desc=f"Epoch {epoch + 1} Training"):
            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probabilities = torch.sigmoid(outputs)

            prediction = (probabilities >= 0.5).float()

            correct += (prediction == labels).sum().item()

            total += labels.size(0)

    average_loss = running_loss/len(validation_dataloader)

    accuracy = correct/total

    return average_loss, accuracy

        

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    train_dataloader, validation_dataloader = create_dataloader()

    model = create_model()
    checkpoint_name = "models/checkpoint/resnet_18_ai_detector_best.pth.pth"
    
    checkpoint = torch.load(checkpoint_name, map_location=device)
    
    
    
    model.load_state_dict(checkpoint["model_state_dict"])
    

    model.to(device)

   

    for parameters in model.parameters():
        parameters.requires_grad = False

    for parameters in model.layer4.parameters():
        parameters.requires_grad = True

    for parameters in model.fc.parameters():
        parameters.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam([
        {
        "params": model.layer4.parameters(),
        "lr": 1e-5
    },{
        "params": model.fc.parameters(),
        "lr": 1e-3
    }

    ])

    #History
    best_validation_loss = checkpoint["validation_loss"]
    history = {"train_loss": [],
               "validation_loss": [],
               "validation_accuracy": []}
    epochs = 10
    for epoch in range(epochs):
        train_loss = train_one_epoch(model=model, device=device, train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, epoch=epoch)
        
        validation_loss, accuracy = validate(model=model, device=device, validation_dataloader=validation_dataloader, criterion=criterion, epoch=epoch  )

        print(f" Progress {epoch + 1}/{epochs}")

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{accuracy:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(accuracy)

        with open("src/training/history/history.pkl", "wb") as f:
            pickle.dump(history, f)


        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {"epoch": epoch +1,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "train_loss": train_loss,
                 "validation_loss": validation_loss,
                 "validation_accuracy": accuracy},
                 "models/checkpoint/resnet_best_validation.pth")

            print("Best Validation Loss Model saved")

    

            print(f"Best validation loss: {best_validation_loss:.4f}")


        torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy
        },
        "models/checkpoint/last_checkpoint.pth")
        print("\nLast checkpoint savec.")


if __name__ == "__main__":
    main()