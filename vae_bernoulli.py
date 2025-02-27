# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
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
    

class MoGPrior(nn.Module):
    def __init__(self, M, num_components=10):
        """
        Define a Mixture of Gaussians (MoG) prior with `num_components` Gaussians.

        Parameters:
        - M: [int] Dimension of the latent space.
        - num_components: [int] Number of Gaussian components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M  # Latent space dimensionality
        self.num_components = num_components  # Number of Gaussians in the mixture
        # Define the means of each Gaussian component (initialized randomly)
        self.means = nn.Parameter(torch.randn(num_components, M)*2, requires_grad=True)
        # Define the standard deviations of each Gaussian component (fixed small variance)
        self.stds = nn.Parameter(torch.ones(num_components, M), requires_grad=False)
        # Define mixture weights (uniform initially)
        self.mixing_logits = nn.Parameter(torch.ones(num_components), requires_grad=True)

    def forward(self):
        """
        Return the Mixture of Gaussians prior distribution.

        Returns:
        - prior: [torch.distributions.Distribution] Mixture of Gaussians prior.
        """
        # Create categorical distribution for mixture weights
        mixing_distribution = td.Categorical(logits=self.mixing_logits)
        # Create independent Gaussian distributions for each component
        component_distribution = td.Independent(td.Normal(loc=self.means, scale=self.stds), 1)
        # Create the Mixture of Gaussians prior
        mixture_distribution = td.MixtureSameFamily(mixing_distribution, component_distribution)

        return mixture_distribution

class VampPrior(nn.Module):
    def __init__(self, pseudo_input_shape, K, encoder, use_learnable_weights=False):
        """
        Define the VampPrior distribution.
        """
        super(VampPrior, self).__init__()
        self.K = K
        self.encoder = encoder
        self.pseudo_inputs = nn.Parameter(torch.randn(K, *pseudo_input_shape), requires_grad=True)
        self.use_learnable_weights = use_learnable_weights
        if self.use_learnable_weights:
            self.logits = nn.Parameter(torch.zeros(K), requires_grad=True)
        else:
            self.register_buffer('logits', torch.zeros(K))
        
    def forward(self):
        # Pass pseudo-inputs through the encoder to obtain Gaussian parameters.
        output = self.encoder(self.pseudo_inputs)
        # Note: assuming output shape (K, 2*M)
        M = output.shape[1] // 2
        mean = output[:, :M]
        logvar = output[:, M:]
        std = torch.exp(0.5 * logvar)
        component_distribution = td.Independent(td.Normal(loc=mean, scale=std), 1)
        mixing_distribution = td.Categorical(logits=self.logits)
        return td.MixtureSameFamily(mixing_distribution, component_distribution)

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

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes a tensor of dimension `(batch_size, M)`
           (where M is the latent dimension) as input and outputs a tensor of 
           dimension (batch_size, feature_dim1, feature_dim2) representing the mean.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # We parameterize the log variance (or log standard deviation) to ensure positivity.
        # Here we initialize it to zeros so that std = exp(0.5*0) = 1.0 initially.
        self.log_std = nn.Parameter(torch.zeros(28, 28))

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.
        
        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        # The network outputs the mean of the Gaussian.
        mu = self.decoder_net(z)
        # Compute the standard deviation; using exp(0.5*log_std) ensures it is positive.
        std = torch.exp(0.5 * self.log_std)
        # Return a Gaussian distribution with diagonal covariance.
        return td.Independent(td.Normal(loc=mu, scale=std), 2)



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
        # self.decoder(z).log_prob(x) is the log-likelihood has shape = btach_size
        # same for kl divergence
        if isinstance(self.prior(), td.MixtureSameFamily):
            log_likelihood = self.decoder(z).log_prob(x).mean(0)  # Average over samples

            # Compute Monte Carlo KL divergence estimation
            log_qzx = q.log_prob(z)  # Log probability of sampled z under q(z|x)
            log_pz = self.prior().log_prob(z)  # Log probability of sampled z under MoG prior
            kl_divergence = (log_qzx - log_pz).mean(0)  # Average over samples

            # Compute ELBO
            elbo = log_likelihood - kl_divergence
        else:
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
    progress_bar = tqdm(range(total_steps), desc="Training", miniters= int(total_steps/100), maxinterval=float("inf"))

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", refresh=False)
            progress_bar.update()


def plot_aggregate_posterior(model, data_loader, device, latent_dim):
    """
    Plots the aggregate posterior in 2D, color-coded by the true digit labels.
    """
    model.eval()
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for x, y in data_loader:
            z = model.encoder(x.to(device)).rsample()
            all_latents.append(z.cpu())
            all_labels.append(y)
    all_latents = np.vstack(all_latents)
    all_labels = np.hstack(all_labels)

    # If latent_dim > 2, project to 2D using PCA
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(all_latents)
    else:
        latents_2d = all_latents

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=all_labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title("Aggregate Posterior (Color-Coded by Digit Label)")
    plt.xlabel("Dim 1" if latent_dim == 2 else "PCA 1")
    plt.ylabel("Dim 2" if latent_dim == 2 else "PCA 2")
    plt.tight_layout()
    plt.show()


def plot_prior_contours(model, data_loader, device, latent_dim):
    """
    Plots the 2D filled contours of the prior distribution along with black dots
    from the aggregated posterior (unlabeled).
    """
    model.eval()
    # Gather latents (no labels needed here)
    all_latents = []
    with torch.no_grad():
        i = 0
        for x, _ in data_loader:
            # if i > 5:
            #     break
            z = model.encoder(x.to(device)).sample()
            all_latents.append(z.cpu())
            i += 1

    all_latents = np.vstack(all_latents)

    # Project to 2D if needed
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(all_latents)
    else:
        latents_2d = all_latents

    # Determine plot limits
    x_min, x_max = latents_2d[:, 0].min(), latents_2d[:, 0].max()
    y_min, y_max = latents_2d[:, 1].min(), latents_2d[:, 1].max()
    dx = 0.1 * (x_max - x_min)
    dy = 0.1 * (y_max - y_min)
    x_min -= dx; x_max += dx
    y_min -= dy; y_max += dy

    # Mesh grid for prior evaluation
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # If PCA used, invert for original latent space
    if latent_dim > 2:
        grid_points_orig = pca.inverse_transform(grid_points)
    else:
        grid_points_orig = grid_points

    # Evaluate prior
    grid_tensor = torch.tensor(grid_points_orig, dtype=torch.float32, device=device)
    with torch.no_grad():
        log_probs = model.prior().log_prob(grid_tensor)

        probs = torch.exp(log_probs).cpu().numpy().reshape(xx.shape)

    

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], s=10, c='k')
    
    plt.contourf(xx, yy, probs, levels=20, cmap="viridis", alpha=0.8)
    # plt colorbar
    plt.colorbar(label="Prior Density")
    # Plot the means of the prior components if they exist
    if hasattr(model.prior, 'means'):
        means = model.prior.means.detach().cpu().numpy()
        if latent_dim > 2:
            means = pca.transform(means)
        plt.scatter(means[:, 0], means[:, 1], s=50, c='red', marker='x', label='Prior Means')
        plt.legend()

    plt.title("Prior Contours with Aggregate Posterior Samples")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

def evaluate(model: VAE, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            loss = model(x)
            total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'multi_run'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'vamp'], help='prior distribution to use (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST ( 'data/' , train = True , download = True ,
    #     transform = transforms.Compose([
    #     transforms.ToTensor (),
    #     transforms.Lambda(lambda x : x.squeeze())
    #     ])), batch_size=args.batch_size, shuffle=True)
    
    # mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST ( 'data/' , train = False , download = True ,
    #     transform = transforms.Compose([
    #     transforms.ToTensor (),
    #     transforms.Lambda(lambda x : x.squeeze())
    #     ])), batch_size=args.batch_size, shuffle=True)

    M = args.latent_dim


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

        # Define prior distribution
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MoGPrior(M, num_components=5)
    elif args.prior == 'vamp':
        prior = VampPrior((28, 28), 10, encoder_net, use_learnable_weights=True)

    print(type(prior))
    

    # Define VAE model
    decoder = GaussianDecoder(decoder_net)
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

    elif args.mode == 'test':
        # Load the trained model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Compute ELBO on test set
        elbo = 0.0
        with torch.no_grad():
            for x in mnist_test_loader:
                x = x[0].to(device)
                elbo += model.elbo(x).sum().item()
        elbo /= len(mnist_test_loader.dataset)
        print(f"ELBO on test set: {elbo:.4f}")

        # Plot aggregate posterior
        plot_aggregate_posterior(model, mnist_test_loader, device, args.latent_dim)

        # Plot prior contours
        plot_prior_contours(model, mnist_test_loader, device, args.latent_dim)


    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'multi_run':
        # For robust evaluation, run multiple training runs and record the test loss.
        runs = 5
        losses = []
        for run in range(runs):
            print(f"Run {run+1}/{runs}")
            # (Re-)initialize model and prior for each run.
            if args.prior == 'gaussian':
                prior = GaussianPrior(M)
            elif args.prior == 'mog':
                prior = MoGPrior(M, 5)
            elif args.prior == 'vamp':
                prior = VampPrior((28, 28), 500, encoder=encoder_net, use_learnable_weights=True)
            model = VAE(prior, decoder, encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(model, optimizer, mnist_train_loader, args.epochs, device)
            test_loss = evaluate(model, mnist_test_loader, device)
            print(f"Test loss (negative ELBO): {test_loss:.4f}")
            losses.append(test_loss)
        losses = np.array(losses)
        print(f"Mean test loss: {losses.mean():.4f} ± {losses.std():.4f}")
        with open('test_loss.txt', 'a') as f:
            f.write(f"{args.prior} over {runs} runs: Mean = {losses.mean():.4f}, Std = {losses.std():.4f}\n")
