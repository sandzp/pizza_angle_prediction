import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class KeypointDetector(nn.Module):
    def __init__(self, use_dropout=False):
        '''
        Initialize model architecture
        '''
        super(KeypointDetector, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        if self.use_dropout:
            self.dropout1 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        if self.use_dropout:
            self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * 56 * 56, 1000)
        self.fc2 = nn.Linear(1000, 4) 

    def forward(self, x):
        '''
        Define forward pass
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.use_dropout:
            x = self.dropout1(x)
        x = x.view(-1, 64 * 56 * 56)  
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs, device):
        '''
        Training loop. Prints average loss after each epoch. 

        Inputs:
        train_loader = the DataLoader class containing preprocessed data
        criterion = loss function
        optimizer = optimizer
        num_epochs = Number of training epochs
        device = GPU or CPU
        '''
        average_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs = data['image'].to(device)
                labels = data['keypoints'].to(device)
                optimizer.zero_grad()
                outputs = self(inputs) 
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)

            average_losses.append(average_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        print('Finished Training')

        self.plot_loss(average_losses, len(average_losses))

    def evaluate_model(self, model, dataloader, criterion, device):
        '''
        Eval loop. Prints average loss for all validation data. 

        Inputs:
        model = the trained model
        train_loader = the DataLoader class containing preprocessed data
        criterion = loss function
        num_epochs = Number of training epochs

        Outputs:
        average loss
        '''
        model.eval() 

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad(): 
            for data in dataloader:
                inputs = data['image'].to(device) 
                labels = data['keypoints'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        # print(f'Average Loss: {avg_loss:.4f}')

        return avg_loss

    def plot_loss(self, losses, num_epochs):
        '''
        Plots average training loss curve

        Inputs:
        losses = list of average losses
        num_epochs = length of losses
        '''
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, num_epochs + 1), losses, marker='o')
        plt.title(f"Average Training Loss over {num_epochs} Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.show()

    def predict(self, model, input_tensor):
        '''
        Uses model to predict keypoints

        Inputs:
        model = trained model
        input_tensor = the image on which prediction is to be run
        '''
        input_tensor = torch.tensor(input_tensor).unsqueeze(0)  
        model.eval()
        with torch.no_grad(): 
            prediction = model(input_tensor)
        output = prediction.numpy()
        return output.squeeze(0)


