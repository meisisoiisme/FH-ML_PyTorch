from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

def train_with_scheduler_and_early_stopping(self):
    scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    patience = 5
    early_stop_counter = 0

    for epoch in range(self.num_epochs):
        self.model.train()

        for inputs, labels, landmarks in self.train_loader:
            inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, landmarks in self.val_loader:
                inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_val_loss}, Validation Accuracy: {correct/total}')

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
