from circuitsvis.tokens import colored_tokens
from transformers import PreTrainedTokenizer

from deception_detection.data.base import DialogueDataset
from deception_detection.experiment import Experiment
from deception_detection.tokenized_data import TokenizedDataset
from deception_detection.utils import process_str_tokens


def get_examples_for_unpaired_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset: DialogueDataset,
    n: int = 1,
):
    """
    Tokenizes a dataset and displays n examples with the token scores highlighted.
    Returns an HTML snippet.
    """
    toks = TokenizedDataset.from_dataset(dataset, tokenizer)

    html_snippet = ""

    for i in range(min(n, len(toks.str_tokens))):
        tok = process_str_tokens(toks.str_tokens[i])

        assert toks.detection_mask is not None
        mask = toks.detection_mask[i][-len(tok) :].float()

        # Create HTML for honest tokens
        honest_html = (
            str(
                colored_tokens(
                    tok,
                    mask.float(),
                    min_value=0,
                    max_value=1,
                    positive_color="blue",
                    negative_color="white",
                )
            )
            + "</div>"
        )

        # Add to the HTML snippet
        html_snippet += f"<h3>Example {i+1}</h3>"
        html_snippet += f"<p><strong>{honest_html}</p>"

    return html_snippet


def get_examples_for_datasets(
    tokenizer: PreTrainedTokenizer,
    datasets: list[DialogueDataset],
    n: int = 1,
) -> str:
    """
    Tokenizes a list of datasets and displays n examples with the detection mask highlighted.
    Returns a complete html file.
    """
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dataset Examples with Highlighted Detection Mask</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            h1 { color: #333; }
            h2 { color: #666; }
            .example { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Dataset Examples with Highlighted Detection Mask</h1>
        {{content}}
    </body>
    </html>
    """

    all_examples = ""

    for dataset in datasets:
        all_examples += f"<h2>Dataset {dataset.name}, n={len(dataset)}</h2>"
        dataset_examples = get_examples_for_unpaired_dataset(tokenizer, dataset, n)
        all_examples += f'<div class="example">{dataset_examples}</div>'

    complete_html = HTML_TEMPLATE.replace("{{content}}", all_examples)

    return complete_html


def save_detection_masks(experiment: Experiment):
    datasets: list[DialogueDataset] = experiment.eval_datasets + experiment.control_datasets
    file_path = experiment.cfg.folder / "detection_masks.html"
    content = get_examples_for_datasets(experiment.tokenizer, datasets)  # type: ignore
    with open(file_path, "w") as f:
        f.write(content)
