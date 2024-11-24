# DiffuSphere - An Object-Oriented Framework for Image Generation using Diffusion Models and their Variants.

**DiffuSphere is a cutting-edge framework** designed to streamline and enhance the process of **image generation using diffusion models**. Built with industry-standard coding methodologies, DiffuSphere ensures high scalability, maintainability, and efficient bug tracking, making it suitable for research and production environments. 

- The framework adopts a modular and object-oriented architecture, enabling developers to extend or customize its components effortlessly while maintaining code clarity and robustness.

With **DiffuSphere**, **zero coding is required to generate images using the latest and most powerful diffusion model variants**. The framework is designed to work out-of-the-box, allowing users to leverage advanced diffusion models without writing a single line of code. 

- Furthermore, integrating new models into the framework is incredibly straightforward, as 95% of the required code infrastructure is already in place. This allows researchers and developers to focus on innovation rather than boilerplate code.

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

##### ---- Main Scripts ---->

1. **`call_methods.py`**: Handles the creation of datasets, networks, and models dynamically based on user specifications.
2. **`train.py`**: Main training script for running and managing the model training loops.


##### ---- `data` Directory ---->

1. **`datasets.py`**: Defines the base class for datasets, including data loading and preprocessing functionalities.
2. **`mnist.py`**: Contains dataset classes for handling MNIST training and testing datasets.
3. **`topographies.py`**: Implements the `BiologicalObservation` dataset class for working with biological images and topographical data.


##### ---- `model` Directory ---->

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


##### ---- `option` Directory ---->
1. **`base_options.py`**: Defines the base configuration options for datasets, models, and training parameters.
2. **`train_options.py`**: Extends base options with training-specific configurations such as learning rate and optimizer settings.


##### ---- `utils` Directory ---->
1. **`images_utils.py`**: Provides utility functions for image transformations, including resizing and normalization.
2. **`utils.py`**: General utilities such as setting seeds for reproducibility and directory management.


## How to Use

Follow these steps to set up and run **DiffuSphere** for training and image generation:

#### 1. Install Requirements
Ensure you have all the dependencies installed. Run the following command:
```
pip install -r requirements.txt
```
#### 2. Clone the Repository
Clone the DiffuSphere repository to your local machine:
```
git clone https://github.com/yourusername/DiffuSphere.git
cd DiffuSphere
```

#### 3. Create a Repository for Loading Data
Prepare a directory structure for your data:

Place your training data (e.g., images, labels) in a new directory, preferably located outside the main DiffuSphere repository (to handle large datasets effectively). For example:
```
mkdir -p /path/to/large_dataset_repo
```

Configure the dataset path in the base_options.py file:

Example: Modify the dataset path in base_options.py
```
--image_folder = "/path/to/large_dataset_repo/images"
--label_path = "/path/to/large_dataset_repo/labels.csv"
```

Also **DiffuSphere is highly configurable via flags** and script modifications:

Flags in base_options.py and train_options.py:
Dataset parameters like --dataset_name, image size, batch size, and more.
Training parameters such as learning rate, number of epochs, optimizer type.

#### 4. Start Training
Run the training process:
```
python train.py
```

#### Optional: Advanced Flag Management with train.sh

1. If you have multiple flags to modify or want a streamlined way to manage configurations:

Open launch and edit the train.sh file:

Example: 
```
python train.py \
--images_folder '../../Dataset/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed' \
--label_path '../../Dataset/biology_data/TopoChip/MacrophageWithClass.csv' \
--dataset_name 'biological' \
--n_epochs 40000 \
--img_size 64 \
--batch_size 32 \
--num_workers 4 \

# Add the parameters and values accordingly
```

2. Run the script:
```
./train.sh
```

#### 5. Predict

Readme for predict will be updated soon. Apologies!


