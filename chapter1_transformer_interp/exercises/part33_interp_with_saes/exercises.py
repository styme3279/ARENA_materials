# %%

import gc
import itertools
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias

import circuitsvis as cv
import einops
import pandas as pd
import plotly.express as px
import requests
import torch as t
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from IPython.display import HTML, IFrame, display
from jaxtyping import Float, Int
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from sae_lens import (
    SAE,
    ActivationsStore,
    GatedTrainingSAEConfig,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
)
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")


def _get_hook_layer(sae: SAE) -> int:
    """Extract the layer number from an SAE's hook name (e.g. 'blocks.7.hook_resid_pre' → 7)."""
    return int(sae.cfg.metadata.hook_name.split(".")[1])


# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part33_interp_with_saes"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part33_interp_with_saes.tests as tests
import part33_interp_with_saes.utils as utils

MAIN = __name__ == "__main__"

# %%
