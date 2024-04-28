### GANWrapper Documentation

#### Installation
```bash
pip install torch torchvision scikit-learn matplotlib
```

#### Description
The `GANWrapper` class provides a high-level interface for training, generating samples, evaluating, and performing inference with Generative Adversarial Networks (GANs).

#### Functions

| Function                | Description                                           | Inputs                                                   | Outputs                                               |
|-------------------------|-------------------------------------------------------|----------------------------------------------------------|-------------------------------------------------------|
| `__init__`              | Initialize GANWrapper with generator and discriminator models. | `generator_model (nn.Module)`: Generator model <br> `discriminator_model (nn.Module)`: Discriminator model | None                                                  |
| `train`                 | Train the GAN using separate trainers for generator and discriminator. | `training_args (dict)`: Training arguments <br> `train_dataset (torch.utils.data.Dataset)`: Training dataset <br> `val_dataset (torch.utils.data.Dataset, optional)`: Validation dataset <br> `checkpoint_dir (str, optional)`: Directory to save checkpoints | Training logs                                         |
| `save_checkpoint`       | Save a checkpoint of the generator and discriminator models. | `epoch (int)`: Current epoch <br> `checkpoint_dir (str)`: Directory to save the checkpoint | None                                                  |
| `save_model`            | Save a PyTorch model.                                 | `model (torch.nn.Module)`: PyTorch model to save <br> `model_path (str)`: File path to save the model | None                                                  |
| `load_model`            | Load a pre-trained generator model from a checkpoint file. | `checkpoint_path (str)`: Path to the checkpoint file     | None                                                  |
| `generate_samples`      | Generate samples using the trained generator model.   | `num_samples (int)`: Number of samples to generate       | Generated samples                                     |
| `evaluate`              | Evaluate the discriminator model on the evaluation dataset. | `eval_dataset (torch.utils.data.Dataset)`: Evaluation dataset | Evaluation metrics                                    |
| `infer`                 | Perform inference on input data using the generator model. | `data`: Input data for inference                        | Generated output                                      |
| `visualize_sample_images` | Visualize generated samples as images.                | `samples (torch.Tensor)`: Generated samples <br> `num_rows (int, optional)`: Number of rows in the visualization grid <br> `num_cols (int, optional)`: Number of columns in the visualization grid | None                                                  |

#### Example Usage
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ganwrapper.ganwrapper import GANWrapper

# Define Generator and Discriminator models
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Define training parameters
training_args = {
    'num_epochs': 40,
    'batch_size': 64,
    'generator_optimizer': optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    'discriminator_optimizer': optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
}

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

# Split dataset into train and validation
train_dataset, val_dataset = train_test_split(mnist_dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)

# Initialize GANWrapper
gan_wrapper = GANWrapper(generator_model=generator, discriminator_model=discriminator)

# Train the GAN
logs = gan_wrapper.train(training_args, train_loader, val_dataset=val_dataset, checkpoint_dir="./checkpoints")

# Load a pre-trained generator model from a checkpoint
checkpoint_path = 'checkpoints/gan_checkpoint_epoch_39.pt'
gan_wrapper.load_model(checkpoint_path)

# Generate samples
num_samples = 30
generated_samples = gan_wrapper.generate_samples(num_samples)

gan_wrapper.visualize_sample_images(generated_samples)
```