# RAVE-Latent Diffusion (Flex'ed)
Generate new latent codes for RAVE with Denoising Diffusion Probabilistic models...with a few more options. 

Author: Moisés Horta Valenzuela / [`𝔥𝔢𝔵𝔬𝔯𝔠𝔦𝔰𝔪𝔬𝔰`](https://twitter.com/hexorcismos). Year: 2023\
Forked and updated by Martin Heinze | [`marts~`](https://martstil.de). Year: 2025

---

## Changes in this fork: 

- __model_config.py__ (new and experimental!)
  - architecture presets for training diffusion models on different latent lengths (and smaller datasets)
- __preprocess.py__
  - added shorter '--latent_length' options to preprocess.py
- __train.py__
  - fixed gradient accumulation 
  - fixed scheduler state mismatch
  - changed '--finetune' flag logic (action='store_true')
  - added model architecture selection (from model_config.py) in order to train on smaller context windows via '--latent_length' flag
  - added noise augmentation ('--noise_std' flag)
  - added break on validation loss not improving for x epochs ('--patience' flag)
  - added flags for learning rate and weight decay ('--lr', '--weight_decay')
  - added flags for '--num_workers'
  - added storing of rave_dims and latent_length to checkpoints (for generate.py)
- __generate.py__
  - fixed seed randomization bug (when '--num > 1')
  - added model architecture retrieval from model_config.py
  - added latent length retrieval from checkpoint
    - '--latent_length' flag now optional (backwards compatibility)
  - changed '--lerp' flag logic (action='store_true')
  - added '--latent_scale' flag to normalize latent codes to a definable value
  - added '--skip_normalize' to skip latent code normalization altogether

---

## Install

Create and activate a new conda environment e.g.:

```bash
conda create -n rave-latent-diffusion python=3.9
conda activate rave-latent-diffusion
```

Clone this repository, run pip install on requirements.txt

```bash
git clone https://github.com/devstermarts/RAVE-Latent-Diffusion-Flex-ed.git
cd RAVE-Latent-Diffusion-Flex-ed
pip install .
```

## Preprocessing
*See __/scripts/preprocess.py__ for all flags and arguments.*

```bash
rldf-preprocess \
--rave_model "/path/to/your/pretrained/rave/stereo_model.ts" \
--audio_folder "/path/to/your/audio/dataset" \
--latent_length 256 \ # 512, 1024, 2048, 4096, 8192, 16384
--latent_folder "/path/to/save/encoded/rave/latents"
```

## Training
*See __/scripts/train.py__ for all flags and arguments.*

```bash
rldf-train \
--name your-run-name \
--latent_length 256 \ # 512, 1024, 2048, 4096, 8192, 16384. Must match latent length in preprocessing
--latent_folder "/path/to/saved/encoded/rave/latents" \
--save_out_path "/path/to/save/rave-latent-diffusion/checkpoints"
```

## Generation
*See __/scripts/generate.py__ for all flags and arguments.*

```bash
rldf-generate \
--model_path "/path/to/trained/rave-latent-diffusion/model.pt" \
--rave_model "/path/to/your/pretrained/rave/model.ts" \
--diffusion_steps 100 \
--temperature 0.9 \
--seed 123456789 \
--output_path "/path/to/save/generated/audio" \
--latent_mult 2
```

### Spherical interpolation between generated RAVE latents

```bash
rldf-generate \
--model_path "/path/to/trained/rave-latent-diffusion/model.pt" \
--rave_model "/path/to/your/pretrained/rave/model.ts" \
--diffusion_steps 100 \
--temperature 0.9 \
--seed_a 123456789 \
--seed_b 987654321 \
--lerp \
--lerp_factor 1.0 \
--output_path "/path/to/save/generated/audio" \
--latent_mult 2
```

---
## Credits

- Original repository [RAVE-Latent-Diffusion](https://github.com/moiseshorta/RAVE-Latent-Diffusion)
  - This code builds on the work done at acids-ircam with [`RAVE (Caillon, 2022)`](https://arxiv.org/abs/2111.05011).
  - The denoising diffusion model is based on the open-source [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch) library.
  - Many thanks to Zach Evans from Harmon.ai for helping debug the code.
