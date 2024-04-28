import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


class GANWrapper:
    def __init__(self, generator_model=None, discriminator_model=None):
        """
        Initialize GANWrapper with generator and discriminator models.

        Args:
            generator_model (nn.Module): Generator model.
            discriminator_model (nn.Module): Discriminator model.
        """
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_batch(self, dataset, idx, batch_size):
        # Fetch batch from dataset
        start_idx = idx * batch_size
        end_idx = (idx + 1) * batch_size
        return dataset[start_idx:end_idx]

    def train(self, training_args, train_dataset, val_dataset=None, checkpoint_dir=None):
        """
        Train the GAN using separate trainers for generator and discriminator.

        Args:
            training_args (dict): Training arguments.
            train_dataset (torch.utils.data.Dataset): Training dataset.
            val_dataset (torch.utils.data.Dataset, optional): Validation dataset. Defaults to None.
            checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to None.

        Returns:
            dict: Training logs.
        """
        # Unpack training arguments
        num_epochs = training_args['num_epochs']
        batch_size = training_args['batch_size']
        generator_optimizer = training_args['generator_optimizer']
        discriminator_optimizer = training_args['discriminator_optimizer']

        # Initialize optimizers
        generator_optimizer = optim.Adam(self.generator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discriminator_optimizer = optim.Adam(self.discriminator_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()

        # Training loop
        for epoch in range(num_epochs):
            # Training mode
            self.generator_model.train()
            self.discriminator_model.train()
            
            epoch_loss = 0.0

            for i, (real_images, _) in enumerate(train_dataset):
                # Move data to device
                real_images = real_images.to(self.device)

                # Generate fake images
                z = torch.randn(batch_size, self.generator_model.latent_dim, device=self.device)
                fake_images = self.generator_model(z)

                # Train discriminator
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                discriminator_optimizer.zero_grad()

                real_output = self.discriminator_model(real_images)
                fake_output = self.discriminator_model(fake_images.detach())  # Detach to prevent gradients flowing to generator

                real_loss = criterion(real_output, real_labels)
                fake_loss = criterion(fake_output, fake_labels)
                d_loss = real_loss + fake_loss

                d_loss.backward()
                discriminator_optimizer.step()

                # Train generator
                generator_optimizer.zero_grad()

                fake_output = self.discriminator_model(fake_images)
                g_loss = criterion(fake_output, real_labels)

                g_loss.backward()
                generator_optimizer.step()

                # Accumulate epoch loss
                epoch_loss += (d_loss.item() + g_loss.item()) / 2

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataset)

            # Print training progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}")

            # Save checkpoint if checkpoint_dir is provided
            if checkpoint_dir:
                self.save_checkpoint(epoch, checkpoint_dir)

        print("Training complete!")

    def save_checkpoint(self, epoch, checkpoint_dir):
        """
        Save a checkpoint of the generator and discriminator models.

        Args:
            epoch (int): Current epoch.
            checkpoint_dir (str): Directory to save the checkpoint.
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'gan_checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator_model.state_dict(),
            'discriminator_state_dict': self.discriminator_model.state_dict(),
        }, checkpoint_path)

        print(f"Checkpoint saved at {checkpoint_path}")

    def save_model(self, model, model_path):
        """
        Save a PyTorch model.

        Args:
            model (torch.nn.Module): PyTorch model to save.
            model_path (str): File path to save the model.
        """
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, checkpoint_path):
        """
        Load a pre-trained generator model from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            None
        """
        if self.generator_model is None:
            raise ValueError("Generator model is not provided.")

        # Load the generator model state_dict from the checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.generator_model.load_state_dict(checkpoint['generator_state_dict'])       

    def generate_samples(self, num_samples):
        """
        Generate samples using the trained generator model.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        # Generate random noise
        noise = torch.randn(num_samples, self.generator_model.latent_dim, device=self.device)

        # Generate samples
        generated_samples = self.generator_model(noise)

        return generated_samples

    def evaluate(self, eval_dataset):
        """
        Evaluate the discriminator model on the evaluation dataset.

        Args:
            eval_dataset (torch.utils.data.Dataset): Evaluation dataset.

        Returns:
            dict: Evaluation metrics.
        """
        # Evaluate the discriminator model on the evaluation dataset
        dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.discriminator_model(inputs)
                predictions = (outputs > 0.5).float()  # Thresholding at 0.5 for binary classification
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return {"accuracy": accuracy}

    def infer(self, data):
        """
        Perform inference on input data using the generator model.

        Args:
            data: Input data for inference.

        Returns:
            torch.Tensor: Generated output.
        """
        # Perform inference using the generator model
        data = data.to(self.device)
        with torch.no_grad():
            output = self.generator_model(data)
        return output

    def visualize_sample_images(self, samples, num_rows=4, num_cols=4):
        """
        Visualize generated samples as images.

        Args:
            samples (torch.Tensor): Generated samples.
            num_rows (int, optional): Number of rows in the visualization grid. Defaults to 4.
            num_cols (int, optional): Number of columns in the visualization grid. Defaults to 4.
        """
        num_samples = min(len(samples), num_rows * num_cols)  # Ensure num_samples doesn't exceed total samples
        samples_detached = samples.detach()[:num_samples]  # Slice the samples tensor to the required length

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_samples:  # Check if the index is within bounds
                sample = samples_detached[i].cpu().numpy()  # Convert tensor to numpy array
                if sample.ndim == 1:  # If the sample is flattened, reshape it to a 2D image
                    sample = sample.reshape((28, 28))  # Adjust shape as per your image dimensions
                ax.imshow(sample, cmap='gray')  # Use gray colormap for grayscale images
                ax.axis("off")
            else:
                ax.axis("off")  # Hide empty subplot

        plt.tight_layout()
        plt.show()

