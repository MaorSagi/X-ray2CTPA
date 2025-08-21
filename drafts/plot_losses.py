import re
import json
import matplotlib.pyplot as plt

# Path to your log file
log_file_path = 'outputs/2025-08-16/17-55-57/train_ddpm.log'  # Change this to the correct file name

# Initialize loss containers
total_loss = []
lpips_loss = []
disc_loss = []
cls_loss = []
l1_loss = []
gen_loss=[]
cls_train_accuracy=[]
cls_train_total_accuracy=[]
cls_val_total_accuracy=[]
cls_val_accuracy=[]
# Read and parse the log file
with open(log_file_path, 'r') as f:
    for line in f:
        match = re.search(r'\{.*\}', line)
        if match:
            try:
                loss_dict = json.loads(match.group().replace("'", '"'))
                total_loss.append(loss_dict.get('total_loss', 0))
                lpips_loss.append(loss_dict.get('lpips_loss', 0))
                disc_loss.append(loss_dict.get('disc_loss', 0))
                cls_loss.append(loss_dict.get('cls_loss', 0))
                l1_loss.append(loss_dict.get('l1_loss', 0))
                gen_loss.append(loss_dict.get('gen_loss', 0))
                cls_train_accuracy.append(loss_dict.get('cls_train_accuracy', 0))
                cls_train_total_accuracy.append(loss_dict.get('cls_train_total_accuracy', 0))
                cls_val_total_accuracy.append(loss_dict.get('cls_val_total_accuracy', 0))
                cls_val_accuracy.append(loss_dict.get('cls_val_accuracy', 0))
            except json.JSONDecodeError:
                continue



# Plotting
plt.figure(figsize=(12, 6))
# plt.plot(total_loss, label='Total Loss')
plt.plot(lpips_loss, label='LPIPS Loss')
# plt.plot(disc_loss, label='Disc Loss')
# plt.plot(cls_loss, label='Cls Loss')

plt.xlabel('Step (Log Line Index)')
plt.ylabel('Loss Value')
plt.title('Loss Metrics Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()