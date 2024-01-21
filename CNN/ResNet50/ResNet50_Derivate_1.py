class DifferentResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(DifferentResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),  # Adjusted output features
            nn.ReLU(),
            nn.Dropout(0.3),  # Adjusted dropout rate
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
