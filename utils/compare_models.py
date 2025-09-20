import torch
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))

model_paths_dict = {
        "resnet18" : "../results/resnet/epochs_loss.pt",
        "se_net" : "../results/se_net/epochs_loss.pt",
        }

for model_name, path in model_paths_dict.items():
    try:
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, torch.Tensor):
            metric_tensor = data
        else:
            print(f"Formato no reconocido en {model_name}: {path}")
            continue

        plt.plot(metric_tensor, label=model_name)
    
    except Exception as e:
        print(f"Error al cargar {model_name} desde {path}: {e}")

plt.xlabel('Época')
plt.ylabel('Loss')
plt.title("ResNet18 vs SENet")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
