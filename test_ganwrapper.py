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