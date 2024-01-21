import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class CustomTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        # Additional attributes for tracking metrics
        self.history = {'train_loss': [], 'val_loss': [], 'accuracy': []}

    def initialize_model(self):
        # Custom initialization logic for the model and optimizer
        # Example: self.model = ...

    def load_pretrained_model(self, model_path):
        # Load a pre-trained model
        # Example: self.model.load_state_dict(torch.load(model_path))

    def set_learning_rate_scheduler(self, step_size=10, gamma=0.1):
        # Set up a learning rate scheduler
        scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return scheduler

    def early_stopping(self, patience=5):
        # Early stopping logic
        best_val_loss = float('inf')
        early_stop_counter = 0

        def should_stop(epoch, val_loss):
            nonlocal best_val_loss, early_stop_counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                return False
            else:
                early_stop_counter += 1
                return early_stop_counter >= patience

        return should_stop

    def calculate_metrics(self, outputs, labels):
        # Calculate additional metrics
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

        precision = precision_score(labels.cpu().numpy(), predicted_labels, average='weighted')
        recall = recall_score(labels.cpu().numpy(), predicted_labels, average='weighted')
        f1 = f1_score(labels.cpu().numpy(), predicted_labels, average='weighted')
        accuracy = accuracy_score(labels.cpu().numpy(), predicted_labels)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

        return metrics

    def train_with_scheduler_and_early_stopping(self):
        scheduler = self.set_learning_rate_scheduler()
        stop_condition = self.early_stopping()

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels, landmarks in self.train_loader:
                inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_train_loss)

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
            accuracy = correct / total
            self.history['val_loss'].append(avg_val_loss)
            self.history['accuracy'].append(accuracy)

            print(f'Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Validation Accuracy: {accuracy}')

            scheduler.step()
            if stop_condition(epoch, avg_val_loss):
                print(f'Early stopping at epoch {epoch+1}')
                break
   
    
    def train_with_mixup_augmentation(self, alpha=0.2):
        # Implementation of the mixup augmentation training strategy
        for epoch in range(self.num_epochs):
            self.model.train()

            for inputs, labels, landmarks in self.train_loader:
                inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)

                self.optimizer.zero_grad()

                # Mixup augmentation
                lam = np.random.beta(alpha, alpha)
                inputs_mix, labels_mix = self.mixup(inputs, labels, lam)

                outputs = self.model(inputs_mix)
                loss = self.criterion(outputs, labels_mix)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            validation_loss, accuracy = self.run_validation_epoch()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {validation_loss}, Validation Accuracy: {accuracy}')

    def train_with_transfer_learning(self):
        # Implementation of the transfer learning training strategy
        for epoch in range(self.num_epochs):
            self.model.train()

            for inputs, labels, landmarks in self.train_loader:
                inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)

                self.optimizer.zero_grad()
                features = self.model.features(inputs)
                outputs = self.model.classifier(features.view(features.size(0), -1))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            validation_loss, accuracy = self.run_validation_epoch()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {validation_loss}, Validation Accuracy: {accuracy}')

    def train(self, counter, experiment_details=None):
        if counter == 1:
            self.train_with_scheduler_and_early_stopping()
        elif counter == 2:
            self.train_with_mixup_augmentation()
        elif counter == 3:
            self.train_with_transfer_learning()
        # Add more conditions for other training functions

        if experiment_details:
            self.log_experiment_details(experiment_details)


    def run_training_epoch(self):
        self.model.train()
        train_loss = 0.0

        for inputs, labels, landmarks in self.train_loader:
            inputs, labels, landmarks = inputs.to(self.device), labels.to(self.device), landmarks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def run_validation_epoch(self):
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

        validation_loss = val_loss / len(self.val_loader)
        accuracy = correct / total

        return validation_loss, accuracy

    def initialize_tensorboard_logging(self, log_dir='./logs'):
        # Initialize TensorBoard logging
        writer = SummaryWriter(log_dir)
        return writer

    def save_model(self, save_path='model.pth'):
        # Save the trained model
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path='model.pth'):
        # Load a saved model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def perform_inference(self, input_data):
        # Perform inference on new data
        # Example: outputs = self.model(input_data)
        # return outputs

    def visualize_training_curves(self):
        # Visualize training curves (loss and accuracy)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
