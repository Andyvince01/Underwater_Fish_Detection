# 🐠 FUnIE-GAN

from pathlib import Path

from .PyTorch.nets.funiegan import GeneratorFunieGAN

__all__ = "GeneratorFunieGAN"

# Path to the pre-trained Funie-GAN model
FUNIEGAN_DIR = Path(__file__).parent.joinpath("PyTorch/models/funie_generator.pth")