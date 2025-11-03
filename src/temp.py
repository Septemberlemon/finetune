from datasets import load_dataset
from torchvision import transforms as T
from matplotlib import pyplot as plt


# mnist_dataset = load_dataset('ylecun/mnist')
# train_mnist_dataset = mnist_dataset['train']
# augmentations = T.Compose([
#     T.RandomHorizontalFlip(),
#     T.RandomRotation(15),
#     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     T.ToTensor(), # 将 PIL Image 转换为 PyTorch Tensor (0-1 a 范围内)
#     T.Resize((224, 224)),
#     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化到 -1 到 1
# ])
# processed_train_mnist_dataset = train_mnist_dataset.set_transform()
