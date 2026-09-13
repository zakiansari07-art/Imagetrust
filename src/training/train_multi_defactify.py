import torch
import torch.nn as nn
from data.dataloader_defactify import create_generator_task_dataloader
from models.detector import create_defactify_generator_model
import pickle
from tqdm import tqdm
from pathlib import Path
import json

project_root = Path.cwd()

path_name = project_root.parent

best_val_cp = "multi_task_defactify_best_validation.pth"

last_cp = "multi_task_defactify_last_checkpoint.pth"

hist_file_name = "history_multi_task.json"


def train_one_epoch(model, device, train_dataloader, criterion, optimizer, epoch):

    model.train()
    
    running_loss = 0

    for images, generators in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training"):
        images = images.to(device)
        generators = generators.to(device).long()

        outputs = model(images)

       
        loss = criterion(outputs, generators)

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
        for images, generators in tqdm(validation_dataloader, desc=f"Epoch {epoch + 1} Validation"):
            images = images.to(device)
            generators = generators.to(device).long()


            outputs = model(images)
            

            loss = criterion(outputs, generators)

            running_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
                                          

            prediction = torch.argmax(probabilities, dim=1)

            correct += (prediction == generators).sum().item()

            total += generators.size(0)

    average_loss = running_loss/len(validation_dataloader)

    accuracy = correct/total

    return average_loss, accuracy

        

def main(epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    train_dataloader, validation_dataloader = create_generator_task_dataloader()

    model = create_defactify_generator_model()

    model.to(device=device, non_blocking=True)


    for parameters in model.parameters():
        parameters.requires_grad = False

    for parameters in model.layer4.parameters():
        parameters.requires_grad = True

    for parameters in model.fc.parameters():
        parameters.requires_grad = True
    

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {
        "params": model.layer4.parameters(),
        "lr": 1e-4
    },{
        "params": model.fc.parameters(),
        "lr": 1e-3
    }

    ])
    
    
    checkpoint_name = Path(f"models/checkpoint/{last_cp}")
   
    if checkpoint_name.exists():

        checkpoint = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        best_validation_loss = checkpoint["best_validation_loss"]

        start_epoch = checkpoint["epoch"]
    else:

        best_validation_loss = float('inf')

        start_epoch = 0

 


    history_path = Path(f"src/training/multiclass_task_history/{hist_file_name}")
   
    if history_path.exists():
        with open(history_path, "r") as file:
            history_checkpoint =  json.load(file)
        history = { "epochs": history_checkpoint["epochs"],
                    "train_loss": history_checkpoint["train_loss"],
                    "validation_loss": history_checkpoint["validation_loss"],
                    "validation_accuracy":history_checkpoint["validation_accuracy"]}
    else:
        
        history = {"epochs": [],
                    "train_loss": [],
                    "validation_loss": [],
                    "validation_accuracy": []}
       
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_one_epoch(model=model, device=device, train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, epoch=epoch)
        
        validation_loss, accuracy = validate(model=model, device=device, validation_dataloader=validation_dataloader, criterion=criterion, epoch=epoch )

        

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
        history["epochs"].append((epoch+1))
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(accuracy)


        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {"epoch": epoch +1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": accuracy
                
                },
                f"models/checkpoint/{best_val_cp}")

            print("Best Validation Loss Model saved")

    

            print(f"Best validation loss: {best_validation_loss:.4f}")

        
        torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy,
            "best_validation_loss": best_validation_loss
        },
        f"models/checkpoint/{last_cp}")
        print("\nLast checkpoint saved.")


        with open(f"src/training/multiclass_task_history/{hist_file_name}", "w") as file:
                            json.dump(history, file, indent=4)
    
if __name__ == "__main__":
    main(epochs=4)