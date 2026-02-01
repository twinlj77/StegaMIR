# SIREN with Latent Modulations for Information Hiding

## Overview
This repository contains an implementation of SIREN (Sinusoidal Representation Networks) with FiLM (Feature-wise Linear Modulation) generated from a latent vector. The code is particularly tailored for information hiding (steganography) tasks, where secret messages can be embedded into the model's parameters. The structure and key components are adapted from the Functa repository.

## Key Features
- **Latent Vector-based Modulation**: Modulation parameters are generated from a compact latent vector, offering flexibility in representation.
- **Flexible FiLM Modulation**: Supports scaling, shifting, or both operations on features.
- **Meta-SGD Support**: Integrates Meta-SGD optimizer for learning adaptive learning rates.
- **Sine Activation with Scalable Frequency**: The [w0](file://D:\experiment\functa(point%20cloud)\functa\function_reps.py#L41-L41) parameter in `sin(w0 * x)` controls the frequency of the sine activation.
- **Fixed Message Extractor**: The `MLPExtractor` is a crucial component for steganography, designed to reliably extract hidden messages without needing to be retrained or transmitted alongside the carrier model.
- **Configurable Network Architecture**: Easily adjust network depth, width, and other hyperparameters.

## Setup
To set up a Python virtual environment with the required dependencies, run:

```shell
# create virtual environment
python3 -m venv /tmp/functa_venv
source /tmp/functa_venv/bin/activate
# update pip and install dependencies
pip3 install --upgrade pip setuptools wheel
pip3 install jax jaxlib haiku numpy matplotlib pillow dill
Add repository to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd）
```

## Usage Examples

### 1. Basic Model Initialization and Forward Pass

python 
from function_reps import LatentModulatedSiren 
import jax 
import jax.numpy as jnp

Initialize model
model = LatentModulatedSiren( 
width=256, 
depth=5,
out_channels=3, 
latent_dim=64, 
layer_sizes=(256, 512),
w0=1.0, 
modulate_scale=True, 
modulate_shift=True 
)

Example coordinates (e.g., for a 64x64 image)
coords = jnp.random.uniform(-1, 1, (64, 64, 2))

Initialize parameters
rng = jax.random.PRNGKey(42) 
params = model.init(rng, coords)

Forward pass to generate an image
output_image = model.apply(params, None, coords) # output_image shape: (64, 64, 3)

### 2. Message Embedding and Extraction Workflow
(This is a conceptual example; actual embedding would involve training the model with a specific loss to encode the message into params)


python 
from embed_fnn import MLPExtractor, generate_bit_string
Assume 'params' from above now contains an embedded secret message in its modulation parts

For demonstration, let's say we've embedded a message during training.

Initialize the fixed message extractor

The extractor's architecture is fixed and known to both sender and receiver

message_length = 100 # Length of the embedded bit string 
extractor = MLPExtractor( input_dim=model.modulation_dim, # This would be derived from model's modulation parameters size 
output_dim=message_length, 
hidden_dim=128, 
num_layers=3 
) 
extractor_params = extractor.init(rng, jnp.ones((model.modulation_dim,))) 
# Dummy input for init

Extract the message from the model's parameters

In practice, you'd extract the relevant modulation parameters from 'params'
and pass them to the extractor.

embedded_modulation_params = ... # Extract from params
extracted_bits = extractor.apply(extractor_params, None, 
embedded_modulation_params)
Generate a dummy bit string for reference
dummy_bits = generate_bit_string(message_length) 
print(f"Reference Bit String: {dummy_bits}")

For a complete end-to-end example of message embedding and extraction, refer to the `modulation_em.py` script.

## Core Components
- **Sine Activation**: Applies a scaled sine transform (`sin(w0 * x)`).
- **FiLM Modulation**: Applies `out = scale * in + shift` to features.
- **Latent Vector**: A trainable vector that conditions the generation of modulation parameters.
- **LatentToModulation**: An MLP that decodes the latent vector into a set of FiLM modulation parameters for the SIREN network.
- **MLPExtractor (Fixed Message Extractor)**: A critical component for steganography. It is a pre-defined, fixed-structure neural network that takes modulation parameters (or a specific subset thereof) from a [LatentModulatedSiren](file://D:\experiment\functa(point%20cloud)\functa\function_reps.py#L376-L511) model as input and outputs the extracted binary message. Its fixed nature ensures it doesn't need to be transmitted, only its architecture needs to be agreed upon.
- **MetaSGDLrs**: A module for storing and managing learnable learning rates for Meta-SGD optimization.

## Training Script for Information Hiding
The `modulation_em.py` script demonstrates an end-to-end workflow for training a [LatentModulatedSiren](file://D:\experiment\functa(point%20cloud)\functa\function_reps.py#L376-L511) model to embed a secret message into its parameters and then using the `MLPExtractor` to retrieve it.


```shell
Example usage (modify parameters as needed)
python3 -m modulation_em 
--mod_dim=64 
--pretrained_weights_dir=pretrained_weights 
--save_to_dir=modulation_data 
--message_length=32
```

## References
This implementation is based on concepts from:
- "From data to functa: Your data point is a function and you can treat it like one" (Dupont et al., 2022).
- Code structure and modulation techniques are adapted from the Functa repository.
- SIREN: "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann et al., 2020).
- The specific application to information hiding using a fixed extractor is inspired by the "Fixed Neural Network Steganography" paradigm.

## License
Apache License 2.0 - see LICENSE file for details.
