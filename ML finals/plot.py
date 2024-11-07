import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件加载数据
df = pd.read_csv("results.csv", header=None, names=["train_losses", "test_losses", "deep_train_losses", "deep_test_losses"])

first_half = df["train_losses"].iloc[0:1000]  # 前 1000 个数据
second_half = df["train_losses"].iloc[1000:2000]  # 后 1000 个数据
third_half = df["train_losses"].iloc[2000:3000]  # 后 1000 个数据
fourth_half = df["train_losses"].iloc[3000:4000]  # 后 1000 个数据
# do the same for test losses
first_half_test = df["test_losses"].iloc[0:1000]  # 前 1000 个数据
second_half_test = df["test_losses"].iloc[1000:2000]  # 后 1000 个数据
third_half_test = df["test_losses"].iloc[2000:3000]  # 后 1000 个数据
fourth_half_test = df["test_losses"].iloc[3000:4000]  # 后 1000 个数据

# do the same for deep train losses
first_half_deep = df["deep_train_losses"].iloc[0:1000]  # 前 1000 个数据
second_half_deep = df["deep_train_losses"].iloc[1000:2000]  # 后 1000 个数据
third_half_deep = df["deep_train_losses"].iloc[2000:3000]  # 后 1000 个数据
fourth_half_deep = df["deep_train_losses"].iloc[3000:4000]  # 后 1000 个数据
# do the same for deep test losses
first_half_deep_test = df["deep_test_losses"].iloc[0:1000]  # 前 1000 个数据
second_half_deep_test = df["deep_test_losses"].iloc[1000:2000]  # 后 1000 个数据
third_half_deep_test = df["deep_test_losses"].iloc[2000:3000]  # 后 1000 个数据
fourth_half_deep_test = df["deep_test_losses"].iloc[3000:4000]  # 后 1000 个数据

# 绘制 Shallow MLP Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, 1001), first_half, label="Batch Size = 64")
plt.plot(range(1, 1001), second_half, label="Batch Size = 128")
plt.plot(range(1, 1001), third_half, label="Batch Size = 256")
plt.plot(range(1, 1001), fourth_half, label="Batch Size = 1000")
plt.title("Shallow MLP Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# 绘制 Shallow MLP Test Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, 1001), first_half_test, label="Batch Size = 64")
plt.plot(range(1, 1001), second_half_test, label="Batch Size = 128")
plt.plot(range(1, 1001), third_half_test, label="Batch Size = 256")
plt.plot(range(1, 1001), fourth_half_test, label="Batch Size = 1000")
plt.title("Shallow MLP Test Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# 绘制 Deep MLP Training Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, 1001), first_half_deep, label="Batch Size = 64")
plt.plot(range(1, 1001), second_half_deep, label="Batch Size = 128")
plt.plot(range(1, 1001), third_half_deep, label="Batch Size = 256")
plt.plot(range(1, 1001), fourth_half_deep, label="Batch Size = 1000")
plt.title("Deep MLP Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# 绘制 Deep MLP Test Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, 1001), first_half_deep_test, label="Batch Size = 64")
plt.plot(range(1, 1001), second_half_deep_test, label="Batch Size = 128")
plt.plot(range(1, 1001), third_half_deep_test, label="Batch Size = 256")
plt.plot(range(1, 1001), fourth_half_deep_test, label="Batch Size = 1000")
plt.title("Deep MLP Test Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()