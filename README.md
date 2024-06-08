## README for FashionMNIST GAN Modifications

### The Use of Any Third-Party Resources
- **Code:** https://github.com/diegoalejogm/gans/blob/master/2.%20DC-GAN%20PyTorch.ipynb
- **Dataset:** [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Video:** [YouTube Tutorial on GANs](https://www.youtube.com/watch?v=OljTVUVzPpM)

### Motivation

The motivation for this project stems from the desire to explore the capabilities of Generative Adversarial Networks (GANs) in generating diverse and high-quality images of fashion items. Previously, GANs have shown impressive results in generating realistic images in various domains, such as handwritten digits (MNIST) and other natural images. By adapting a GAN model to the FashionMNIST dataset, this project aims to push the boundaries of GAN applications to the fashion domain, which involves more complex and varied patterns compared to traditional datasets like MNIST. The design and development process involved modifying the existing GAN architecture to suit the characteristics of FashionMNIST, testing, and iteratively improving the model to achieve better image generation quality. The evaluation process included visual inspection of generated images and quantitative metrics to assess the performance and diversity of the generated fashion items.

### Overview

This project adapts an existing GAN model originally designed for the MNIST dataset to generate images from the FashionMNIST dataset. FashionMNIST consists of grayscale images of fashion items, unlike the handwritten digits of MNIST. This adaptation involves modifying several parts of the code to handle the different characteristics of the FashionMNIST dataset, which is essential to accurately generate fashion item images that are representative of the actual dataset.



### Specific Code Changes

1. **Dataset Change:** 
   The dataset was changed from MNIST, which typically contains grayscale images of handwritten digits, to FashionMNIST, which contains grayscale images of various fashion items.

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

4. **Number of Test Samples Increase:**
   The number of test samples was increased from 16 to 32 to provide a more extensive evaluation of the model's performance.
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

5. **Batch Size Adjustment:**
   The batch size for training was reduced from 100 to 50 to allow more frequent updates per epoch, which is intended to improve the model's responsiveness to data variations and enhance learning efficiency.
   - **Original Code:**
     ```python
     batch_size = 100
     data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
     ```
   - **Modified Code:**
     ```python
     batch_size = 50
     data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
     ```
## Impact of Changes

### Training Update Frequency
By reducing the batch size from 100 to 50, the number of updates per epoch doubles. This can lead to more detailed weight adjustments and potentially faster convergence on smaller batch sizes.

### Computational Resource Utilization
Reducing the batch size generally reduces memory usage, which can be beneficial for running on systems with limited resources.

### Model Performance
Smaller batches often provide a more noisy estimate of the gradient, which can help in escaping local minima during training.

## Conclusion

The adjustments made to the MNIST data loading configuration, including the batch size change, are aimed to balance efficient computation with effective



     

**Using ChatGPT for Assistance:** During the modification of some data and code structures, I utilized ChatGPT to get assistance. ChatGPT helped in understanding the necessary changes needed to adapt the model from MNIST to FashionMNIST, ensuring the model parameters and data handling are correctly adjusted for the new dataset. Additionally, I used ChatGPT to write the README and to help me check for some bugs, such as fixing a color channel mismatch in the dataset at first.I give ChatGPT my original code and modified code to let him write the code.






