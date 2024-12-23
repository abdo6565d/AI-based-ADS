# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the results from the specified CSV file
results_df = pd.read_csv(r"C:\Users\habib\PycharmProjects\yolov5directory\yolov5-master\runs\train\exp44\results.csv")

# Strip whitespace from column names
results_df.columns = results_df.columns.str.strip()

# Print cleaned column names to verify
print("Cleaned column names:")
print(results_df.columns)

# Extracting the relevant loss columns
train_box_losses = results_df['train/box_loss'].tolist()
train_cls_losses = results_df['train/cls_loss'].tolist()
val_box_losses = results_df['val/box_loss'].tolist()
val_cls_losses = results_df['val/cls_loss'].tolist()

# Number of epochs
epochs = len(train_box_losses)

# Plotting the training and validation losses
plt.figure(figsize=(12, 6))

# Plot for box losses
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_box_losses, label='Training Box Loss', marker='o')
plt.plot(range(1, epochs + 1), val_box_losses, label='Validation Box Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Box Loss Over Epochs')
plt.legend()
plt.grid()

# Plot for class losses
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_cls_losses, label='Training Class Loss', marker='o')
plt.plot(range(1, epochs + 1), val_cls_losses, label='Validation Class Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Class Loss Over Epochs')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Print final results
print("\nFinal Results:")
print(f"Final Training Box Loss: {train_box_losses[-1]:.4f}")
print(f"Final Validation Box Loss: {val_box_losses[-1]:.4f}")
print(f"Final Training Class Loss: {train_cls_losses[-1]:.4f}")
print(f"Final Validation Class Loss: {val_cls_losses[-1]:.4f}")
