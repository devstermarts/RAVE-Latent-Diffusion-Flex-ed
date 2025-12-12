#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moisés Horta Valenzuela / @hexorcismos
#### Year: 2023
#### ----------
#### Updates in this fork: Martin Heinze
#### Year: 2025

# Todos/ ideas:
# - Add starting/ending point and/or timeframe for slerp


import argparse
import os
import random

import numpy as np
import soundfile as sf
import torch
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from model_config import get_architecture

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_latent_dim(rave):
    return rave.decode_params[0].item()


# Parse the input arguments for the script.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RAVE latents using diffusion model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained diffusion model checkpoint.",
    )
    parser.add_argument(
        "--rave_model",
        type=str,
        required=True,
        help="Path to the pretrained RAVE model (.ts).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        choices=[44100, 48000],
        help="Sample rate for generated audio. Should match samplerate of RAVE model.",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of steps for denoising diffusion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--latent_length",
        type=int,
        default=None,
        choices=[256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Length of RAVE latents. Only used if checkpoint doesn't contain this info.",
    )
    parser.add_argument(
        "--length_mult",
        type=int,
        default=1,
        help="Multiplies the duration of output by default model window.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to where you want to save the audio file.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of audio to generate.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="output",
        help="Name of audio to generate.",
    )
    parser.add_argument(
        "--lerp",
        action="store_true",
        help="Interpolate between two seeds.",
    )
    parser.add_argument(
        "--lerp_factor",
        type=float,
        default=1.0,
        help="Interpolating factor between two seeds.",
    )
    parser.add_argument(
        "--seed_a",
        type=int,
        default=None,
        help="Starting seed for interpolation.",
    )
    parser.add_argument(
        "--seed_b",
        type=int,
        default=None,
        help="Ending seed for interpolation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature of the random noise before diffusion.",
    )
    parser.add_argument(
        "--latent_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to normalized latents before RAVE decoding.",
    )
    parser.add_argument(
        "--skip_normalize",
        action="store_true",
        help="Skip latent normalization before RAVE decoding.",
    )
    return parser.parse_args()


def slerp(val, low, high):
    omega = torch.acos(
        (
            low
            / torch.norm(low, dim=2, keepdim=True)
            * high
            / torch.norm(high, dim=2, keepdim=True)
        )
        .sum(dim=2, keepdim=True)
        .clamp(-1, 1)
    )
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (
        torch.sin(val * omega) / so
    ) * high
    return res


# Latent normalization helper function
def normalize_latents(diff, scale=1.0):
    diff_mean = diff.mean()
    diff_std = diff.std()
    normalized = (diff - diff_mean) / diff_std
    return normalized * scale


# Generate the audio using the provided models and settings.
def generate_audio(model, rave, args, seed):
    with torch.no_grad():
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        rave_dims = get_latent_dim(rave)
        z_length = args.latent_length * args.length_mult

        noise = torch.randn(1, rave_dims, z_length).to(device)
        noise = noise * args.temperature

        print(
            f"Generating {z_length} latent codes with Diffusion model:",
            os.path.basename(args.model_path),
        )
        print("Decoding using RAVE Model:", os.path.basename(args.rave_model))
        print("Seed:", seed)

        model.eval()

        ### GENERATING WITH .PT FILE
        diff = model.sample(noise, num_steps=args.diffusion_steps, show_progress=True)

        if not args.skip_normalize:
            diff = normalize_latents(diff, scale=args.latent_scale)

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = rave.decode(diff)
        y = y.reshape(-1).detach().numpy()

        if rave.stereo:
            y_l = y[: len(y) // 2]
            y_r = y[len(y) // 2 :]
            y = np.stack((y_l, y_r), axis=-1)

        path = f"{args.output_path}/{args.name}_{args.seed}.wav"
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)


# Generate audio by slerping between two diffusion generated RAVE latents.
def interpolate_seeds(model, rave, args, seed):
    with torch.no_grad():
        torch.manual_seed(seed)  # Remove, no effect?

        z_length = args.latent_length * args.length_mult

        rave_dims = get_latent_dim(rave)

        torch.manual_seed(args.seed_a)
        noise1 = torch.randn(1, rave_dims, z_length).to(device) * args.temperature
        torch.manual_seed(args.seed_b)
        noise2 = torch.randn(1, rave_dims, z_length).to(device) * args.temperature

        print(
            f"Generating {z_length} latent codes with Diffusion model:",
            os.path.basename(args.model_path),
        )
        print("Decoding using RAVE Model:", os.path.basename(args.rave_model))
        print("Interpolating with factor", args.lerp_factor)
        print("Seed A:", args.seed_a)
        print("Seed B:", args.seed_b)

        model.eval()

        diff1 = model.sample(noise1, num_steps=args.diffusion_steps, show_progress=True)
        diff2 = model.sample(noise2, num_steps=args.diffusion_steps, show_progress=True)
        diff = slerp(
            torch.linspace(0.0, args.lerp_factor, z_length).to(device), diff1, diff2
        )
        if not args.skip_normalize:
            diff = normalize_latents(diff, scale=args.latent_scale)

        rave = rave.cpu()
        diff = diff.cpu()
        print("Decoding using RAVE Model...")
        y = rave.decode(diff)
        y = y.reshape(-1).detach().numpy()

        if rave.stereo:
            y_l = y[: len(y) // 2]
            y_r = y[len(y) // 2 :]
            y = np.stack((y_l, y_r), axis=-1)

        path = f"{args.output_path}/{args.name}_{args.seed_a}_{args.seed_b}_slerp.wav"  # Check what happens when seed_a/seed_b=None
        print(f"Writing {path}")
        sf.write(path, y, args.sample_rate)


# Main function sets up the models and generates the audio.
def main():
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Get latent_length: CLI argument takes priority, but warn if mismatched
    if args.latent_length is not None:
        latent_length = args.latent_length
        if "latent_length" in checkpoint:
            if checkpoint["latent_length"] != latent_length:
                raise ValueError(
                    f"Cannot use --latent_length={latent_length}. "
                    f"Checkpoint was trained with latent_length={checkpoint['latent_length']}. "
                    f"Model architecture must match the checkpoint."
                )
        else:
            print(
                f"Warning: No information about latent_length in checkpoint. Using --latent_length={latent_length}"
            )
    elif "latent_length" in checkpoint:
        latent_length = checkpoint["latent_length"]
        print(f"Using latent_length={latent_length} from checkpoint")
    else:
        raise ValueError(
            "Checkpoint doesn't contain latent_length and --latent_length not specified. "
            "Please provide --latent_length matching what was used during training."
        )

    args.latent_length = latent_length

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    if args.seed_a is None:
        args.seed_a = random.randint(0, 2**31 - 1)

    if args.seed_b is None:
        args.seed_b = random.randint(0, 2**31 - 1)

    rave = torch.jit.load(args.rave_model).to(device)
    rave_dims = get_latent_dim(rave)

    # Validate RAVE dimensions match checkpoint if available
    if "rave_dims" in checkpoint and checkpoint["rave_dims"] != rave_dims:
        raise ValueError(
            f"RAVE model has {rave_dims} dimensions, but checkpoint was trained with {checkpoint['rave_dims']}"
        )

    if not args.sample_rate:
        msg = "RAVE model doesn't store its sample rate. --sample_rate is required."
        assert hasattr(rave, "sr"), msg
        args.sample_rate = rave.sr

    ### GENERATING WITH .PT FILE DIFFUSION
    arch = get_architecture(latent_length)  # Get architecture for this latent_length

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=rave_dims,
        channels=arch["channels"],
        factors=arch["factors"],
        items=arch["items"],
        attentions=arch["attentions"],
        attention_heads=arch["attention_heads"],
        attention_features=arch["attention_features"],
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if not args.lerp:
        for i in range(args.num):
            seed = args.seed + i
            generate_audio(model, rave, args, seed)
    else:
        for i in range(args.num):
            seed = args.seed + i
            interpolate_seeds(model, rave, args, seed)


if __name__ == "__main__":
    main()
