# DiffuSphere - An Object-Oriented Framework for Image Generation using Diffusion Models and their Variants.

**DiffuSphere is a cutting-edge framework** designed to streamline and enhance the process of **image generation using diffusion models**. Built with industry-standard coding methodologies, DiffuSphere ensures high scalability, maintainability, and efficient bug tracking, making it suitable for research and production environments. 

The framework adopts a modular and object-oriented architecture, enabling developers to extend or customize its components effortlessly while maintaining code clarity and robustness.

With **DiffuSphere**, **zero coding is required to generate images using the latest and most powerful diffusion model variants**. The framework is designed to work out-of-the-box, allowing users to leverage advanced diffusion models without writing a single line of code. 

Furthermore, integrating new models into the framework is incredibly straightforward, as 95% of the required code infrastructure is already in place. This allows researchers and developers to focus on innovation rather than boilerplate code.

### Supported Models

The framework supports the training and evaluation of various diffusion models, including:

1. **DDPM (Denoising Diffusion Probabilistic Models)**:
   - A generative model that learns to iteratively reverse a noise process to generate high-quality samples.

2. **DDPM with EMA (Exponential Moving Average)**:
   - An enhanced DDPM model with weights updated with an EMA for better stability and improved sample quality.

3. **DDPM CFG (Classifier-Free Guidance)**:
   - A variant of DDPM that uses a guidance mechanism without a classifier to better control the generation process by amplifying desired features.

4. **DDPM CFG EMA**:
   - Combines the benefits of CFG with EMA-based weight updates for even more stable training and higher-quality controlled generation.

5. **CFG++ (Classifier-Free Guidance++)**:
   - An advanced version of Classifier-Free Guidance that improves control and diversity in sample generation by refining the guidance mechanism.

6. **CFG++ EMA**:
   - A CFG++ model incorporating EMA to further enhance stability and generation quality, particularly in fine-grained and high-detail samples.

Each of these models provides flexibility for various use cases, balancing control, stability, and sample quality according to the requirements of the task. The modular design of DiffuSphere ensures seamless transitions between models and effortless integration of new variations.


## Project Structure and Overview

#### Main Scripts

1. **`call_methods.py`**: Handles the creation of datasets, networks, and models dynamically based on user specifications.
2. **`train.py`**: Main training script for running and managing the model training loops.

<hr style="border: 0.2px solid #ccc;" />

#### `data` Directory
1. **`datasets.py`**: Defines the base class for datasets, including data loading and preprocessing functionalities.
2. **`mnist.py`**: Contains dataset classes for handling MNIST training and testing datasets.
3. **`topographies.py`**: Implements the `BiologicalObservation` dataset class for working with biological images and topographical data.

<hr style="border: 0; height: 1px; background: #e1e4e8;" />

#### `model` Directory
1. **`attention_block.py`**: Implements attention mechanisms for improving the UNet model's performance.
2. **`ddpm.py`**: Contains the implementation of Denoising Diffusion Probabilistic Models (DDPM) instance and its variants. 
3. **`downsampling_block.py`**: Implements downsampling operations in the UNet model.
4. **`models.py`**: Defines the base class for all models, including training and saving mechanisms.
5. **`networks.py`**: Base class for networks, defining essential methods like forward pass and parameter counting.
6. **`nin_block.py`**: Implements Network-in-Network (NiN) blocks for feature extraction.
7. **`resnet_block.py`**: Defines residual blocks to facilitate deep feature extraction in the UNet.
8. **`timestep_embedding.py`**: Provides a time embedding mechanism for incorporating temporal information.
9. **`unet.py`**: Implements the UNet architecture tailored for diffusion models.
10. **`upsampling_block.py`**: Handles upsampling operations in the UNet.

-  

#### `option` Directory
1. **`base_options.py`**: Defines the base configuration options for datasets, models, and training parameters.
2. **`train_options.py`**: Extends base options with training-specific configurations such as learning rate and optimizer settings.

---

### `utils` Directory
1. **`images_utils.py`**: Provides utility functions for image transformations, including resizing and normalization.
2. **`utils.py`**: General utilities such as setting seeds for reproducibility and directory management.

---

## How to Use
