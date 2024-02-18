# Generative AI Models

Generative AI models are a type of artificial neural network that can generate new, synthetic data. These models learn the underlying structure of the data they were trained on and then use this knowledge to create new, realistic examples.

## Types of Generative AI Models

1. **Variational Autoencoders (VAEs)**: VAEs are a type of generative model that learns to reconstruct input data by mapping it into a lower-dimensional latent space. VAEs are trained using a combination of reconstruction and regularization losses.

2. **Generative Adversarial Networks (GANs)**: GANs consist of two neural networks, a generator, and a discriminator. The generator creates new data, while the discriminator tries to distinguish between real and generated data. The two networks are trained in a competitive manner, with the generator trying to create more realistic data, and the discriminator trying to become better at distinguishing between real and generated data.

3. **Recurrent Neural Networks (RNNs)**: RNNs are a type of neural network that can process sequences of data. They can be used to generate sequences of data, such as text or music.

4. **Transformers**: Transformers are a type of neural network architecture that can process sequences of data in parallel. They have been used to generate text, images, and other types of data.

5. **Deep Convolutional Generative Adversarial Networks (DCGANs)**: DCGANs are a type of GAN that uses convolutional layers instead of fully connected layers. DCGANs are particularly well-suited for generating images.

6. **Generative Pre-trained Transformer (GPT)**: GPT is a type of transformer model that can generate text. It has been pre-trained on a large corpus of text data and then fine-tuned to generate specific types of text, such as news articles or social media posts.

These models have a wide range of applications, including generating new images, music, or text, creating synthetic data for machine learning, and even generating new molecules for drug discovery.

7. https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/ RAGs
## Example

In the code provided, a VAE is being trained on the MNIST dataset of handwritten digits. The VAE learns to reconstruct the input images by mapping them into a lower-dimensional latent space. The loss function combines reconstruction and regularization losses, which are used to train the model.

Here's a simple example of how you might use a pre-trained VAE to generate new images:

# Load a pre-trained VAE
vae = torch.load('vae.pth')
vae.eval()

# Generate a batch of new images
with torch.no_grad():
    z = torch.randn(128, 2).to(device)  # Random latent vectors
    generated_images = vae.decoder(z)

# Display the generated images
import matplotlib.pyplot as plt

_, axes = plt.subplots(4, 4)
for i in range(16):
    axes[i // 4, i % 4].imshow(generated_images[i].cpu().numpy(), cmap='gray')
    axes[i // 4, i % 4].axis('off')
plt.show()


