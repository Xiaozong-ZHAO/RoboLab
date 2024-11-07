import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []
q_real_no_correction = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

num_hidden_units = 32
# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4,num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units,num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)
    

class DeepCorrectorMLP(nn.Module):
    def __init__(self):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4,num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units,num_hidden_units),
            nn.ReLU(),
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)

learning_rate = 0.00001
# Model, Loss, Optimizer
model = MLP()
deep_model = DeepCorrectorMLP()
criterion = nn.MSELoss()
deep_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
deep_optimizer = optim.Adam(deep_model.parameters(), lr=learning_rate)



# Training Loop
epochs = 1000
train_losses = []
deep_train_losses = []
test_losses = []
deep_test_losses = []

start_time = time.time()
for epoch in range(epochs):
    epoch_loss = 0
    # deep_epoch_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        # deep_optimizer.zero_grad()
        output = model(data)
        # deep_output = deep_model(data)
        loss = criterion(output, target)
        # deep_loss = deep_criterion(deep_output, target)
        loss.backward()
        # deep_loss.backward()
        optimizer.step()
        # deep_optimizer.step()
        epoch_loss += loss.item()
        # deep_epoch_loss += deep_loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    # deep_train_losses.append(deep_epoch_loss / len(train_loader))

    # # 评估模型在测试集上的表现
    # model.eval()  # 切换到评估模式，不计算梯度
    # correction_loss = 0
    # deep_correction_loss = 0
    # with torch.no_grad():  # 禁用梯度计算
    #     for data, target in test_loader:
    #         output = model(data)
    #         deep_output = deep_model(data)
    #         correction_loss += criterion(output, target).item()
    #         deep_correction_loss += deep_criterion(deep_output, target).item()
    
    # # 计算平均测试损失
    # test_losses.append(correction_loss / len(test_loader))
    # deep_test_losses.append(deep_correction_loss / len(test_loader))
    
    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')
    #     print(f'Epoch {epoch+1}/{epochs}, Loss: {deep_train_losses[-1]:.6f}')
end_time = time.time()
total_time = end_time - start_time
print(f'Total training time: {total_time:.2f} seconds')

# Testing Phase: Simulate trajectory tracking
q_test = 0
dot_q_test = 0
q_real = []
q_real_corrected = []


# integration with only PD Control
for i in range(len(t)):
    tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
    ddot_q_real = (tau - b * dot_q_test) / m
    dot_q_test += ddot_q_real * dt
    q_test += dot_q_test * dt
    q_real.append(q_test)

q_test = 0
dot_q_test = 0
for i in range(len(t)):
    # Apply MLP correction
    tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
    inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
    correction = model(inputs.unsqueeze(0)).item()
    ddot_q_corrected =(tau - b * dot_q_test + correction) / m
    dot_q_test += ddot_q_corrected * dt
    q_test += dot_q_test * dt
    q_real_corrected.append(q_test)


# loss_data = {
#     "train_losses": train_losses,
#     "test_losses": test_losses,
#     "deep_train_losses": deep_train_losses,
#     "deep_test_losses": deep_test_losses
# }
# results_df = pd.DataFrame(loss_data)
# results_df.to_csv("results.csv", mode='a', index=False, header=False)
# print("Done")


# def moving_average(data, window_size=10):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# plt.figure(figsize=(12,6))
# plt.plot(range(1, epochs + 1), train_losses, label='Shallow MLP Training Loss')
# plt.plot(range(1, epochs + 1), test_losses, label='Shallow MLP Test Loss')
# # plt.plot(range(10, epochs + 1), moving_average(train_losses, window_size=10), label='Shallow Moving Average')
# plt.plot(range(1, epochs + 1), deep_train_losses, label='Deep MLP Training Loss')
# plt.plot(range(1, epochs + 1), deep_test_losses, label='Deep MLP Test Loss')
# # plt.plot(range(10, epochs + 1), moving_average(deep_train_losses, window_size=10), label='Deep Moving Average')
# # set y limit
# # plt.yscale('log')
# plt.title('Loss Convergence Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.grid(True)
# # display total time
# # plt.text(0.5, 0.5, f'Total Time: {total_time:.2f} seconds', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.show()

# print("Done")

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.plot(t, q_target, 'r-', label='Target')
# plt.plot(t, q_real, 'b--', label='PD Only')
# plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
# plt.title('Trajectory Tracking with and without MLP Correction')
# plt.xlabel('Time [s]')
# plt.ylabel('Position')
# plt.legend()
# plt.show()
