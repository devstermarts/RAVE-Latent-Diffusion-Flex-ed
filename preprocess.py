#### RAVE-Latent Diffusion
#### https://github.com/moiseshorta/RAVE-Latent-Diffusion
####
#### Author: Moisés Horta Valenzuela / @hexorcismos
#### Year: 2023
#### ----------
#### Updates in this fork: Martin Heinze
#### Year: 2025

import argparse
import hashlib  # for hashing paths
import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rave_model",
        type=str,
        default="/path/to/rave_model",
        help="Path to the exported (.ts) RAVE model (needs to be one with stereo=true).",
    )
    parser.add_argument(
        "--audio_folder",
        type=str,
        default="/path/to/audio_folder",
        help="Path to the folder containing audio files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        choices=[44100, 48000],
        help="Sample rate of the audio files.",
    )
    parser.add_argument(
        "--latent_length",
        type=int,
        default=1024,
        choices=[256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Length of saved RAVE latents. 1024 equals about 48 seconds at 44100Hz",
    )
    parser.add_argument(
        "--latent_folder",
        type=str,
        default="latents",
        help="Path to the folder where RAVE latent files will be saved.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=["wav", "opus", "mp3", "aac", "flac"],
        help="Extensions to search for in audio_folder",
    )
    return parser.parse_args()


def encode_and_save_latent(
    rave, audio_data, output_base_name, latent_folder, latent_length
):
    with torch.no_grad():
        x = torch.from_numpy(audio_data).reshape(1, 1, -1)

        if device.type != "cpu":
            x = x.to(device)
            rave = rave.to(device)

        z = rave.encode(x)
        print("Encoded into latent", z.shape)

        z_mean = z.mean()
        z_std = z.std()
        z = (z - z_mean) / z_std

        if device.type != "cpu":
            z = z.cpu()

        z = torch.nn.functional.pad(z, (0, latent_length - z.shape[2]))

        z = z.detach().numpy()

        print("Saving latent of shape", z.shape)

        np.save(os.path.join(latent_folder, output_base_name + ".npy"), z)


# from RAVE
def load_audio_chunk(path: str, n_signal: int, sr: int) -> Iterable[np.ndarray]:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "panic",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    chunk = process.stdout.read(n_signal * 2)

    while len(chunk) == n_signal * 2:
        yield chunk
        chunk = process.stdout.read(n_signal * 2)

    process.stdout.close()


def main():
    args = parse_args()

    os.makedirs(args.latent_folder, exist_ok=True)

    rave = torch.jit.load(args.rave_model).to(device)

    crop_samples = args.latent_length * 2048  # 2048 is RAVE encoder downsampling factor

    audio_folder = Path(args.audio_folder)
    audio_files = [f for ext in args.extensions for f in audio_folder.rglob(f"*.{ext}")]

    pbar = tqdm(audio_files)
    for audio_file in pbar:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        # hash dirname to avoid conflicts between same filenames in different dirs
        hasher = hashlib.new("md5")
        hasher.update(os.path.dirname(audio_file).encode())
        dir_hash = hasher.hexdigest()

        chunks = load_audio_chunk(
            os.path.abspath(audio_file), n_signal=crop_samples, sr=args.sample_rate
        )

        pbar.set_description(os.path.relpath(audio_file, args.audio_folder))
        for i, chunk_bytes in enumerate(chunks):
            cropped_data = (
                np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 2**15
            )
            output_file = f"{dir_hash}_{base_name}_part{i:03d}"
            pbar.set_postfix_str(f"chunk:{i}")
            encode_and_save_latent(
                rave, cropped_data, output_file, args.latent_folder, args.latent_length
            )

    print("Done encoding RAVE latents into ", args.latent_folder)


if __name__ == "__main__":
    main()
