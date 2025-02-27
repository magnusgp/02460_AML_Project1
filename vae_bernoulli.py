# Modified code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a mixture of Gaussians prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int] 
           Number of components in the mixture.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.logits = nn.Parameter(torch.ones(self.K), requires_grad=True)
        
    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.MixtureSameFamily(td.Categorical(logits=self.logits), td.Independent(td.Normal(loc=self.means, scale=self.stds), 1))
    
class VampPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a VampPrior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int] 
           Number of components in the mixture.
        """
        super(VampPrior, self).__init__()
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.logits = nn.Parameter(torch.ones(self.K), requires_grad=True)
        
    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.MixtureSameFamily(td.Categorical(logits=self.logits), td.Independent(td.Normal(loc=self.means, scale=self.stds), 1))

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training", miniters=int(18750/100))

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            
def evaluate(model: VAE, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Evaluates the model on the MNIST test data set.

    Args:
        model (VAE): Trained VAE model.
        data_loader (torch.utils.data.DataLoader): Data loader for the MNIST test data.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: Average negative ELBO on the test data set
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            loss = model(x)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def plot_samples(model: VAE, data_loader: torch.utils.data.DataLoader, device: torch.device):
    """Plot samples from the approximate posterior and colour them by their correct class
        label for each datapoint in the test set (i.e., samples from the aggregate posterior).
        Implement it such that you, for latent dimensions larger than two, M > 2, do
        PCA and project the sample onto the first two principal components (e.g., using
        scikit-learn).
        
    Args:
        model (VAE): Trained VAE model.
        data_loader (torch.utils.data.DataLoader): Data loader for the MNIST test data.
        device (torch.device): Device to run the evaluation on.
        
    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        samples = []
        for x in data_loader:
            x = x[0].to(device)
            q = model.encoder(x)
            z = q.sample()
            samples.append(z)
        samples = torch.cat(samples, dim=0).cpu()
        
    if samples.shape[1] > 2:
        from sklearn.decomposition import PCA
        samples = PCA(n_components=2).fit_transform(samples)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], c=mnist_test_loader.dataset.targets, cmap='tab10')
        plt.colorbar()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Samples from the aggregate posterior')
        plt.savefig('samples_aggregate_posterior.png')
        plt.show()
    
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(samples[:, 0], samples[:, 1], c=mnist_test_loader.dataset.targets, cmap='tab10')
        plt.colorbar()
        plt.xlabel('Latent dim 1')
        plt.ylabel('Latent dim 2')
        plt.title('Samples from the aggregate posterior')
        plt.savefig('samples_aggregate_posterior.png')
        plt.show()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'evaluate', 'plot'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'vamp'], help='prior distribution over latent space (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    
    # update paths to include model naming
    args.model = args.model.split('.')[0] + '_' + args.prior + '.pt'
    args.samples = args.samples.split('.')[0] + '_' + args.prior + '.png'

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MixtureOfGaussiansPrior(M, 10)
    elif args.prior == 'vamp':
        prior = VampPrior(M, 10)
    else:
        raise ValueError(f"Unknown prior: {args.prior}")

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)
        
        print(f"Model saved to {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
            
        print(f"Samples saved to {args.samples}")
        
    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        test_loss = evaluate(model, mnist_test_loader, device)
        with open('test_loss.txt', 'a') as f:
            f.write(f"{args.prior}: {test_loss:.4f}\n")
        print(f"Average negative ELBO on test data: {test_loss:.4f}")
        
    elif args.mode == 'plot':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        plot_samples(model, mnist_test_loader, device)
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
