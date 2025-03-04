# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate (an approximation of) the DDPM negative ELBO on a batch of data
        by randomly sampling one time step per example.
        
        Parameters:
        -----------
        x: [torch.Tensor]
            A batch of data of shape (batch_size, D).

        Returns:
        --------
        [torch.Tensor]
            The negative ELBO of shape (batch_size,).
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample a random time step t for each data example in the batch
        t = torch.randint(0, self.T, (batch_size,), device=device)
        
        # Sample one noise vector per example
        epsilon = torch.randn_like(x)
        
        # Get the corresponding cumulative product of alphas for each sampled t
        # Reshape to (batch_size, 1) to allow broadcasting
        alpha_bar_t = self.alpha_cumprod[t].view(batch_size, 1)
        
        sqrt_alpha_bar_t = alpha_bar_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
        
        # Generate the noisy version x_t of x_0 using the sampled t
        x_t = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * epsilon
        
        # Create a time encoding for the network; normalized to [0, 1]
        t_tensor = (t.float() / self.T).unsqueeze(1)
        
        # Get the network's prediction of the noise
        epsilon_theta = self.network(x_t, t_tensor)
        
        # Compute the per-example MSE loss between the true noise and the prediction
        mse = F.mse_loss(epsilon_theta, epsilon, reduction='none')
        per_example_mse = mse.view(batch_size, -1).sum(dim=1)
        
        return per_example_mse



    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            if t == 0:
                z = torch.zeros_like(x_t)
            else:
                z = torch.randn_like(x_t)
            
            # Get alpha_bar_t and sqrt_alpha_bar_t
            alpha_bar_t = self.alpha_cumprod[t]
            alpha_t = self.alpha[t]
            sqrt_alpha_t = alpha_t.sqrt()

            t_tensor = torch.full((shape[0],1), float(t)/self.T, device=self.alpha.device)
            espilon_theta = self.network(x_t, t_tensor)

            alpha_fraction = (1-alpha_t) / (1-alpha_bar_t).sqrt()

            x_t = 1/sqrt_alpha_t * (x_t - alpha_fraction * espilon_theta) + self.beta[t].sqrt() * z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    model.train()

    total_epochs = epochs  # Number of epochs
    progress_bar = tqdm(range(total_epochs), desc="Training", position=0)

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        epoch_loss = 0  # Track loss for the epoch

        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)

            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(data_loader)

        # Update progress bar once per epoch
        progress_bar.set_postfix(loss=f"⠀{avg_loss:12.4f}", epoch=f"{epoch+1}/{epochs}")
        progress_bar.update()

    progress_bar.close()  # Close the progress bar after training


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)
    

def evaluate(model, data_loader, device):
    """
    Evaluate the DDPM model on a test set by computing the average negative ELBO loss.

    Parameters:
    -----------
    model: [DDPM]
        The diffusion model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
        DataLoader for the test dataset.
    device: [torch.device]
        The device on which to perform evaluation.

    Returns:
    --------
    float
        The average negative ELBO loss over the test set.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # If the batch is a tuple/list (e.g., MNIST returns (image, label)), select the image tensor.
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            loss = model.loss(batch)
            # Multiply by batch size to aggregate the loss over all samples.
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss



if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    from unet import Unet

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'sample_mnist', 'multi_run'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--load_pretrained', type=bool, default=False, metavar='V', help='load pretrined model (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    if args.data == 'mnist':
        transform = transforms . Compose ([ transforms.ToTensor(),
        transforms.Lambda(lambda x : x + torch.rand (x.shape ) /255) ,
        transforms.Lambda (lambda x : (x -0.5) *2.0) ,
        transforms.Lambda (lambda x : x.flatten())])
        # Load the MNIST dataset
        def collate_fn(batch):
            images, _ = zip(*batch)  # Unpack and discard labels
            return torch.stack(images)  # Return only images as a batch

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        
    else:
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    # Get the dimension of the dataset
    D = next(iter(train_loader)).shape[1]

    # Define the network
    #num_hidden = 64
    #network = FcNetwork(D, num_hidden)
    network = Unet()

    # calc parameters of the network
    
    print(f"Number of parameters in the network: {sum(p.numel() for p in network.parameters())}")
    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.load_pretrained:
            print("!!! Loaded Pretrained !!!")
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((10000,D))).cpu() 

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        # Plot the density of the toy data and the model samples
        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)
        plt.savefig(args.samples)
        plt.close()
    elif args.mode == 'sample_mnist':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = model.sample((100,D)).cpu() 

        # Transform to range [0, 1]
        samples = (samples + 1) / 2

        # plot 10x10 grid of samples
        samples = samples.view(100, 1, 28, 28)
        save_image(samples, args.samples, nrow=10)


    elif args.mode == 'multi_run':
    # For robust evaluation, run multiple training runs and record the test loss.
        runs = 5
        losses = []
        for run in range(runs):
            print(f"Run {run+1}/{runs}")
            # (Re-)initialize model and prior for each run.
            network = Unet()
            model = DDPM(network, T=T).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # Train model
            train(model, optimizer, train_loader, args.epochs, args.device)
            test_loss = evaluate(model, test_loader, args.device)
            print(f"Test loss (negative ELBO): {test_loss:.4f}")
            losses.append(test_loss)
        losses = np.array(losses)
        print(f"Mean test loss: {losses.mean():.4f} ± {losses.std():.4f}")
        with open('test_loss.txt', 'a') as f:
            f.write(f"{runs} runs: Mean = {losses.mean():.4f}, Std = {losses.std():.4f}\n")
