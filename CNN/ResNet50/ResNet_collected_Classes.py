class ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#######
###
#######

class Derivate_1_ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(Derivate_1_ResNet50, self).__init__()
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

#######
###
#######

class Derivate_2_ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(Derivate_2_ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),  # Adjusted output features
            nn.ReLU(),
            nn.Dropout(0.4),  # Adjusted dropout rate
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
