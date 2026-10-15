import os
import json
from typing import Union, List, Tuple, Optional, Dict, Any
from pkld import pkld
from .attn_funcs import get_avg_attention_matrix, get_vertical_scores
from pytorch_models.model_config import model2layers_heads
import numpy as np
from scipy import stats
from tqdm import tqdm

from utils import sanity_check_sentences, split_solution_keep_spacing


@pkld
def get_vert_score_wrapped(
    text: str,
    sentences: List[str],
    model_name: str = "qwen-15b",
    layer: int = 10,
    head: int = 5,
    proximity_ignore: int = 4,
    control_depth: bool = False,
    score_type: str = "mean",
) -> np.ndarray:
    avg_mat = get_avg_attention_matrix(
        text,
        model_name=model_name,
        layer=layer,
        head=head,
        sentences=sentences,
    )
    assert avg_mat.shape[0] == len(sentences), f"{avg_mat.shape[0]=} {len(sentences)=}"

    vert_scores = get_vertical_scores(
        avg_mat,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
        score_type=score_type,
    )
    assert len(vert_scores) == len(sentences), f"{len(vert_scores)=} {len(sentences)=}"
    return vert_scores


def get_problem_vert_scores(
    layer_head: np.ndarray,
    problem_ci: Tuple[int, bool],
    model_name: str = "qwen-14b",
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    problem_num, is_correct = problem_ci
    text, sentences_w_spacing = get_problem_text_sentences(problem_num, is_correct, model_name)
    layer_head_vert_scores = get_all_heads_vert_scores(
        text,
        sentences_w_spacing,
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
        score_type="mean",
    )
    target_layer_head_vert_scores = layer_head_vert_scores[layer_head[:, 0], layer_head[:, 1], :]
    return target_layer_head_vert_scores


def get_3d_ar_kurtosis(layer_head_vert_scores: np.ndarray) -> np.ndarray:
    layer_head_kurts = stats.kurtosis(
        layer_head_vert_scores, axis=2, fisher=True, bias=True, nan_policy="omit"
    )  # NaNs from the proximity ignorance
    return layer_head_kurts


@pkld
def get_all_heads_vert_scores(
    text: str,
    sentences: List[str],
    model_name: str = "qwen-14b",
    proximity_ignore: int = 4,
    control_depth: bool = False,
    score_type: str = "mean",
) -> np.ndarray:
    layers, heads = model2layers_heads(model_name)
    layer_head_vert_scores = []
    for layer in range(layers):
        layer_scores = []
        for head in range(heads):

            vert_scores = get_vert_score_wrapped(
                text,
                sentences,
                model_name=model_name,
                layer=layer,
                head=head,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
                score_type=score_type,
            )

            layer_scores.append(vert_scores)
        layer_head_vert_scores.append(layer_scores)
    layer_head_vert_scores = np.array(layer_head_vert_scores)
    assert layer_head_vert_scores.shape[-1] == len(sentences)
    return layer_head_vert_scores


def get_top_k_layer_head_kurts(
    layer_head_kurts_mean: np.ndarray,
    top_k: int = 20
) -> np.ndarray:
    kurts_mean_flat = layer_head_kurts_mean.flatten()

    valid_indices = np.where(~np.isnan(kurts_mean_flat))[0]  # indices where it's not NaN
    valid_values = kurts_mean_flat[valid_indices]

    top_k = min(top_k, len(valid_values))  # in case fewer than 20 valid values
    top_indices_in_valid = np.argpartition(valid_values, -top_k)[-top_k:]

    top_indices_in_valid = top_indices_in_valid[np.argsort(-valid_values[top_indices_in_valid])]

    top_flat_indices = valid_indices[top_indices_in_valid]
    return top_flat_indices


@pkld(overwrite=False)
def get_top_k_receiver_heads(
    model_name: str = "qwen-14b",
    top_k: int = 20,
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> np.ndarray:
    resp_layer_head_verts, _ = get_all_problems_vert_scores(
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    resp_layer_head_kurts = []
    for i in range(len(resp_layer_head_verts)):
        layer_head_verts = resp_layer_head_verts[i]
        layer_head_kurts = get_3d_ar_kurtosis(layer_head_verts)
        assert np.sum(np.isnan(layer_head_kurts[1:, :])) == 0
        # if np.isnan(layer_head_kurts).any():
        #     assert np.isnan(layer_head_kurts).all()
        #     continue
        resp_layer_head_kurts.append(layer_head_kurts)
    resp_layer_head_kurts = np.array(resp_layer_head_kurts)
    layer_head_kurts_mean = np.mean(resp_layer_head_kurts, axis=0)
    top_k_layer_head_kurts = get_top_k_layer_head_kurts(layer_head_kurts_mean, top_k)
    layer_head = np.array(np.unravel_index(top_k_layer_head_kurts, layer_head_kurts_mean.shape)).T
    layer_head = layer_head.astype(int)
    assert layer_head.shape[0] == top_k
    assert layer_head.shape[1] == 2
    return layer_head


def get_model_rollouts_root(model_name: str = "qwen-14b") -> str:
    if "qwen" in model_name:
        dir_root = os.path.join(
            "math-rollouts",
            "deepseek-r1-distill-qwen-14b",
            "temperature_0.6_top_p_0.95",
        )
    elif "llama" in model_name:
        dir_root = os.path.join(
            "math-rollouts",
            "deepseek-r1-distill-llama-8b",
            "temperature_0.6_top_p_0.95",
        )
    else:
        raise ValueError(f"Invalid model name, nothing to load: {model_name=}")
    return dir_root


@pkld
def get_problem_text_sentences(
    problem_num: Union[int, str],
    is_correct: bool,
    model_name: str = "qwen-14b"
) -> Tuple[str, List[str]]:
    dir_root = get_model_rollouts_root(model_name)
    if is_correct:
        ci = "correct_base_solution"
    else:
        ci = "incorrect_base_solution"
    dir_ci = os.path.join(dir_root, ci)
    if isinstance(problem_num, int):
        problem_num = f"problem_{problem_num}"
    else:
        assert isinstance(problem_num, str)
        assert problem_num.startswith("problem_")
    dir_problem = os.path.join(dir_ci, problem_num)
    fp_base_solution = os.path.join(dir_problem, "base_solution.json")
    with open(fp_base_solution, "r") as f:
        base_solution = json.load(f)
    text = base_solution["full_cot"]
    sentences_w_spacing = split_solution_keep_spacing(text)
    sanity_check_sentences(sentences_w_spacing, dir_problem, text)
    return text, sentences_w_spacing


@pkld
def get_all_problems_vert_scores(
    model_name: str = "qwen-14b",
    proximity_ignore: int = 4,
    control_depth: bool = False,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:

    dir_root = get_model_rollouts_root(model_name)

    correct_incorrect = ["correct_base_solution", "incorrect_base_solution"]

    response_layer_head_verts = []
    response_idxs = []

    for ci in correct_incorrect:
        if "incorrect" in ci:
            is_correct = 0
        else:
            is_correct = 1
        dir_ci = os.path.join(dir_root, ci)
        problems = os.listdir(dir_ci)
        for idx_problem, problem in enumerate(problems):
            if problem == 'problem_3935': # 13k tokens long, too intense on the RAM/VRAM
                continue
            text, sentences_w_spacing = get_problem_text_sentences(problem, is_correct, model_name)

            # The model will be run.
            # Its sentence-averaged attention weight matrices will be cached.
            # Its vertical attention scores will be cached and returned here.
            # Caching happens automatically through the call stack.
            layer_head_vert_scores = get_all_heads_vert_scores(
                text,
                sentences_w_spacing,
                model_name=model_name,
                proximity_ignore=proximity_ignore,
                control_depth=control_depth,
                score_type="mean",
            )
            assert layer_head_vert_scores.shape[-1] == len(sentences_w_spacing)

            response_layer_head_verts.append(layer_head_vert_scores)
            response_idxs.append((idx_problem, is_correct))
    return response_layer_head_verts, response_idxs


def get_receiver_head_scores(
    top_k_layer_head_kurts: np.ndarray,
    layer_head_verts: np.ndarray
) -> np.ndarray:
    rec_head_scores = []
    for layer, head in top_k_layer_head_kurts:
        rec_head_scores.append(layer_head_verts[layer, head, :])
    rec_head_scores = np.array(rec_head_scores)
    rec_head_score = np.mean(rec_head_scores, axis=0)
    return rec_head_score


@pkld
def get_all_receiver_head_scores(
    model_name: str = "qwen-14b",
    proximity_ignore: int = 4,
    control_depth: bool = False,
    top_k: int = 20,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:

    print("Getting top k layer head kurts")
    top_k_layer_head_kurts = get_top_k_receiver_heads(
        model_name=model_name,
        top_k=top_k,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )
    print("Getting all vert scores")

    response_layer_head_verts, response_idxs = get_all_problems_vert_scores(
        model_name=model_name,
        proximity_ignore=proximity_ignore,
        control_depth=control_depth,
    )

    print("Getting rec scores")
    response_rec_scores = []
    for i in range(len(response_layer_head_verts)):
        layer_head_verts = response_layer_head_verts[i]
        rec_scores = get_receiver_head_scores(top_k_layer_head_kurts, layer_head_verts)
        response_rec_scores.append(rec_scores)
    return response_rec_scores, response_idxs
