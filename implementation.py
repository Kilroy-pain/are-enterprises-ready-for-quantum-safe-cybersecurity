import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QuantumSafeCryptoSimulator:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, x_train, y_train, epochs=100):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(x_train)
            loss = self.loss_fn(predictions, y_train)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, x_test):
        x_test = torch.tensor(x_test, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(x_test)
        return predictions.numpy()

if __name__ == '__main__':
    # Dummy data for testing
    np.random.seed(42)
    x_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, size=(100, 1))  # Binary labels

    x_test = np.random.rand(10, 10)  # 10 test samples

    simulator = QuantumSafeCryptoSimulator(input_dim=10, hidden_dim=20, output_dim=1)
    simulator.train(x_train, y_train, epochs=50)

    predictions = simulator.predict(x_test)
    print("Predictions on test data:")
    print(predictions)