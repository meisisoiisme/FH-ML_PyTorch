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

    print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {val_loss/len(self.val_loader)}, Validation Accuracy: {correct/total}')
