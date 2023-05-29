import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import EyeCenterDataset as eye_dataset

# Define the CNN architecture
class EyeCenterNet(nn.Module):
    def __init__(self):
        super(EyeCenterNet, self).__init__()
        self.features = nn.Sequential(                              # Input size: (1, 3, 40, 40)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),   # size: (1, 64, 40, 40)
            nn.ReLU(inplace=False),                                 # size: (1, 64, 40, 40)
            nn.MaxPool2d(kernel_size=2, stride=2),                  # size: (1, 64, 20, 20)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # size: (1, 128, 20, 20)
            nn.ReLU(inplace=False),                                 # size: (1, 128, 20, 20)
            nn.MaxPool2d(kernel_size=2, stride=2),                  # size: (1, 128, 10, 10)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# size: (1, 256, 10, 10)
            nn.ReLU(inplace=False),                                 # size: (1, 256, 10, 10)
            nn.MaxPool2d(kernel_size=2, stride=2),                  # size: (1, 256, 5, 5)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# size: (1, 512, 5, 5)
            nn.ReLU(inplace=False),                                 # size: (1, 512, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2),                  # size: (1, 512, 2, 2)
        )                                                           #            |   |  |
                                                                    #            |   |  |
        self.dropout = nn.Dropout(p=0.2) # Add dropout layer        #            |   |  |
                                                                    #            |   |  |
        self.fc = nn.Sequential(                                    #            V   V  V
            nn.Linear(512 * 2 * 2, 256), #   <--------------------------       (512 *2 *2)
            nn.ReLU(inplace=False),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 2),  # Adjusted output size to match input size
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout to the flattened features
        x = self.fc(x)
        return x


def try_sample(path_to_folder, name, net):
    unseen_data = Image.open(os.path.join(path_to_folder, name))
    transform = transforms.ToTensor()
    unseen_data = transform(unseen_data).unsqueeze(dim=0)
    center = net(unseen_data).squeeze()
    x = int(center[0])
    y = int(center[1])
    unseen_data[0, 0, y, x] = 1.0
    print("center found at [" + str(x) + ", " + str(y)+"]")

    # Save the tensor as an image
    torchvision.utils.save_image(unseen_data, 'output.png')



def train_model_version_1(path_to_folder):
    # Load dataset
    images = eye_dataset.load_images(path_to_folder)
    centers = eye_dataset.load_eye_centers()

    # Define your training loop
    num_epochs = 250

    # Instantiate the network
    net = EyeCenterNet()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    max_loss = 99999

    # Training loop
    for epoch in range(num_epochs):
        if epoch == 125:
            optimizer = optim.Adam(net.parameters(), lr=0.00005)
        if epoch == 185:
            optimizer = optim.Adam(net.parameters(), lr=0.00001)
        if epoch == 220:
            optimizer = optim.Adam(net.parameters(), lr=0.000005)
        running_loss = 0.0
        for paths, labels in zip(images, centers):
            # Zero the gradients
            optimizer.zero_grad()

            # Get image
            image = Image.open(paths)
            transform = transforms.ToTensor()
            image = transform(image).unsqueeze(dim=0)

            # Forward pass
            outputs = net(image)

            # Compute the loss
            labels = torch.tensor(labels, requires_grad=False, dtype=torch.float32).unsqueeze(dim=0)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(images)}")

        if running_loss < max_loss:
            max_loss = running_loss
            torch.save(net.state_dict(), 'eye_net_weights_version_1.pt')
            print("-> Saved")


def train_model_version_2():
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    # Split dataset into train, validation, and test sets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageFolder("/home/javiserrano/deca-git/deca_testing/TestSamples/actual_eyes_2", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Instantiate the network
    net = EyeCenterNet()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    max_loss = float('inf')
    best_val_loss = float('inf')
    best_model_weights = None

    centers = eye_dataset.load_pupil_centers()

    # Training loop
    num_epochs = 250
    for epoch in range(num_epochs):
        if epoch == 125:
            optimizer = optim.Adam(net.parameters(), lr=0.00005)
        if epoch == 185:
            optimizer = optim.Adam(net.parameters(), lr=0.00001)
        if epoch == 220:
            optimizer = optim.Adam(net.parameters(), lr=0.000005)

        running_loss = 0.0
        net.train()  # Set the model to train mode

        for images, labels in zip(train_loader, centers[:train_size]):
            optimizer.zero_grad()

            # Convert data in correct format
            image = images[0]
            labels = torch.tensor(labels, requires_grad=False, dtype=torch.float32).unsqueeze(dim=0)

            # Forward pass
            outputs = net(image)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print the average loss for the epoch
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")

        # Validate the model
        net.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in zip(val_loader, centers[train_size:val_size+train_size]):
                image = images[0]
                labels = torch.tensor(labels, requires_grad=False, dtype=torch.float32).unsqueeze(dim=0)
                outputs = net(image)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check if the current model performs better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = net.state_dict().copy()
            print("-> Checkpoint created")

        # Early stopping
        if val_loss > max_loss:
            break

    # Load the best model weights
    net.load_state_dict(best_model_weights)

    # Evaluate the model on the test set
    net.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in zip(val_loader, centers[val_size+train_size:]):
            image = images[0]
            labels = torch.tensor(labels, requires_grad=False, dtype=torch.float32).unsqueeze(dim=0)
            outputs = net(image)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss}")

    # Save the best model
    torch.save(net.state_dict(), 'eye_net_weights_version_3.pt')



if __name__ == "__main__":

    abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path_to_folder = os.path.join(abs_deca_dir, 'deca_testing', 'TestSamples', 'actual_eyes')


    # Train model
    #train_model_version_1(path_to_folder)
    #train_model_version_2()

    # Test and load weights


    load_net = EyeCenterNet()
    load_net.load_state_dict(torch.load('eye_net_weights_version_2.pt'))
    # Images 65 through 82 have not been included during training for version_1
    try_sample(path_to_folder,'65.jpg', load_net)
    try_sample(path_to_folder,'66.jpg', load_net)
    try_sample(path_to_folder,'67.jpg', load_net)
    try_sample(path_to_folder,'68.jpg', load_net)
    try_sample(path_to_folder,'69.jpg', load_net)
    try_sample(path_to_folder,'70.jpg', load_net)
    try_sample(path_to_folder,'71.jpg', load_net)
    try_sample(path_to_folder,'72.jpg', load_net)
    try_sample(path_to_folder,'73.jpg', load_net)
    try_sample(path_to_folder,'74.jpg', load_net)
    try_sample(path_to_folder,'75.jpg', load_net)
    try_sample(path_to_folder,'76.jpg', load_net)
    try_sample(path_to_folder,'77.jpg', load_net)
    try_sample(path_to_folder,'78.jpg', load_net)
    try_sample(path_to_folder,'79.jpg', load_net)
    try_sample(path_to_folder,'80.jpg', load_net)
    try_sample(path_to_folder,'81.jpg', load_net)
    try_sample(path_to_folder,'82.jpg', load_net)