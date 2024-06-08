## README for FashionMNIST GAN Modifications

### The Use of Any Third-Party Resources
- **Code:** [GANs in PyTorch for MNIST](https://github.com/diegoalejogm/gans/blob/master/2.%20DC-GAN%20PyTorch-MNIST.ipynb)
- **Dataset:** [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Video:** [YouTube Tutorial on GANs](https://www.youtube.com/watch?v=OljTVUVzPpM)

### Motivation

The motivation for this project stems from the desire to explore the capabilities of Generative Adversarial Networks (GANs) in generating diverse and high-quality images of fashion items. Previously, GANs have shown impressive results in generating realistic images in various domains, such as handwritten digits (MNIST) and other natural images. By adapting a GAN model to the FashionMNIST dataset, this project aims to push the boundaries of GAN applications to the fashion domain, which involves more complex and varied patterns compared to traditional datasets like MNIST. The design and development process involved modifying the existing GAN architecture to suit the characteristics of FashionMNIST, testing, and iteratively improving the model to achieve better image generation quality. The evaluation process included visual inspection of generated images and quantitative metrics to assess the performance and diversity of the generated fashion items.

### Overview

This project adapts an existing GAN model originally designed for the MNIST dataset to generate images from the FashionMNIST dataset. FashionMNIST consists of grayscale images of fashion items, unlike the handwritten digits of MNIST. This adaptation involves modifying several parts of the code to handle the different characteristics of the FashionMNIST dataset, which is essential to accurately generate fashion item images that are representative of the actual dataset.

### Specific Code Changes

1. **Dataset Change:** The dataset was changed from MNIST, which typically contains grayscale images of handwritten digits, to FashionMNIST, which contains grayscale images of various fashion items.

2. **Output Channels Modification in GenerativeNet:**
   - **Original Code:**
     ```python
     self.conv4 = nn.Sequential(
         nn.ConvTranspose2d(
             in_channels=128, out_channels=3, kernel_size=4,
             stride=2, padding=1, bias=False
         )
     )
     self.out = torch.nn.Tanh()
     ```
   - **Modified Code:**
     ```python
     self.conv4 = nn.Sequential(
         nn.ConvTranspose2d(
             in_channels=128, out_channels=1, kernel_size=4,
             stride=2, padding=1, bias=False
         )
     )
     self.out = torch.nn.Tanh()
     ```
   This change ensures that the generated images have the correct format for grayscale images.

3. **Input Channels Modification in DiscriminativeNet:**
   - **Original Code:**
     ```python
     class DiscriminativeNet(torch.nn.Module):
         
         def __init__(self):
             super(DiscriminativeNet, self).__init__()
             
             self.conv1 = nn.Sequential(
                 nn.Conv2d(
                     in_channels=3, out_channels=128, kernel_size=4, 
                     stride=2, padding=1, bias=False
                 ),
                 nn.LeakyReLU(0.2, inplace=True)
             )
     ```
   - **Modified Code:**
     ```python
     class DiscriminativeNet(torch.nn.Module):
         
         def __init__(self):
             super(DiscriminativeNet, self).__init__()
             
             self.conv1 = nn.Sequential(
                 nn.Conv2d(
                     in_channels=1, out_channels=128, kernel_size=4, 
                     stride=2, padding=1, bias=False
                 ),
                 nn.LeakyReLU(0.2, inplace=True)
             )
     ```
   This modification ensures that the discriminator correctly processes the grayscale input images of FashionMNIST.

4. **Number of Test Samples Increase:** The number of test samples was increased from 16 to 32 to provide a more extensive evaluation of the model's performance.
   - **Original Code:**
     ```python
     num_test_samples = 16
     test_noise = noise(num_test_samples)
     ```
   - **Modified Code:**
     ```python
     num_test_samples = 32
     test_noise = noise(num_test_samples)
     ```

5. **Using ChatGPT for Assistance:** During the modification of some data and code structures, I utilized ChatGPT to get assistance. ChatGPT helped in understanding the necessary changes needed to adapt the model from MNIST to FashionMNIST, ensuring the model parameters and data handling are correctly adjusted for the new dataset. Additionally, I used ChatGPT to write the README and to help me check for some bugs, such as fixing a color channel mismatch in the dataset at first.

### Impact of Modifications

Utilizing FashionMNIST instead of MNIST allows the GAN to operate on a different domain of images, testing its ability to generalize and produce fashion-related images. This requires adjustments in model architecture specifically tailored to the type and complexity of images in FashionMNIST. Since FashionMNIST images are grayscale, changing the number of input and output channels in the neural networks is crucial. This ensures that the networks are architecturally compatible with the format of the data, potentially impacting the training dynamics and quality of the generated images. By increasing the number of test samples, we enhance the robustness of the model evaluation. This gives a better understanding of the model's performance across a broader set of data, offering insights into its consistency and reliability in generating diverse fashion items. The assistance from ChatGPT was invaluable in ensuring that the modifications were correctly implemented, thus improving the efficiency and effectiveness of the adaptation process. The above modifications are essential for adapting a GAN model from generating MNIST images to FashionMNIST images, impacting the model's ability to learn and generate high-quality images that faithfully represent the distribution of the FashionMNIST dataset.

### Use of Third-Party Resources

- **Code:** https://github.com/diegoalejogm/gans/blob/master/2.%20DC-GAN%20PyTorch.ipynb
- **Dataset:** [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Video:** [YouTube Tutorial on GANs](https://www.youtube.com/watch?v=OljTVUVzPpM)


