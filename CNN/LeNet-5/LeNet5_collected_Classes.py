class LeNet5(nn.Module):
    def __init__(self, num_classes=6):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120 * 53 * 53, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

###
#
###

class Derivate_1_Net5(nn.Module):
    def __init__(self, num_classes=6):
        super(Derivate_1_Net5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),          # Adjusted initial channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 150, kernel_size=5),        # Increased channels
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(150 * 21 * 21, 128),            # Adjusted input features
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

###
#
###

  class Derivate_2_LeNet5(nn.Module):
    def __init__(self, num_classes=6):
        super(Derivate_2_LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5),  # Fewer initial channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120 * 21 * 21, 64),  # Adjusted input features
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
