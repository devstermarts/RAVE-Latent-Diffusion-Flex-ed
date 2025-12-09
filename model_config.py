# Model architecture configurations for different latent lengths
# The total downsampling factor must be <= latent_length

ARCHITECTURE_PRESETS = {
    256: {
        "channels": [256, 256, 256, 512, 512],
        "factors": [1, 2, 2, 2, 2],  # Total: 16x
        "items": [1, 2, 2, 2, 4],
        "attentions": [0, 0, 0, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    512: {
        "channels": [256, 256, 256, 256, 512, 512],
        "factors": [1, 2, 2, 2, 2, 2],  # Total: 32x
        "items": [1, 2, 2, 2, 2, 4],
        "attentions": [0, 0, 0, 0, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    1024: {
        "channels": [256, 256, 256, 256, 512, 512, 512],
        "factors": [1, 4, 4, 4, 2, 2, 2],  # Total: 512x
        "items": [1, 2, 2, 2, 2, 2, 4],
        "attentions": [0, 0, 0, 0, 1, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    2048: {
        "channels": [256, 256, 256, 256, 512, 512, 512, 768],
        "factors": [1, 4, 4, 4, 2, 2, 2, 2],  # Total: 1024x
        "items": [1, 2, 2, 2, 2, 2, 2, 4],
        "attentions": [0, 0, 0, 0, 0, 1, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    4096: {
        "channels": [256, 256, 256, 256, 512, 512, 512, 768, 768],
        "factors": [1, 4, 4, 4, 2, 2, 2, 2, 2],  # Total: 4096x
        "items": [1, 2, 2, 2, 2, 2, 2, 4, 4],
        "attentions": [0, 0, 0, 0, 0, 1, 1, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    8192: {
        "channels": [256, 256, 256, 256, 512, 512, 512, 768, 768, 768],
        "factors": [1, 4, 4, 4, 2, 2, 2, 2, 2, 2],  # Total: 8192x
        "items": [1, 2, 2, 2, 2, 2, 2, 4, 4, 4],
        "attentions": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
    16384: {
        "channels": [256, 256, 256, 256, 512, 512, 512, 768, 768, 768, 768],
        "factors": [1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2],  # Total: 16384x
        "items": [1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4],
        "attentions": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "attention_heads": 12,
        "attention_features": 64,
    },
}


def get_architecture(latent_length: int) -> dict:
    if latent_length not in ARCHITECTURE_PRESETS:
        raise ValueError(
            f"Unsupported latent_length: {latent_length}. "
            f"Supported values: {list(ARCHITECTURE_PRESETS.keys())}"
        )
    return ARCHITECTURE_PRESETS[latent_length]
