def get_openrouter_model_name(model_name: str) -> str:
    """
    Get the OpenRouter model name for a given model name.
    """
    if model_name == "qwen3/qwen3-30b-a3b":
        return r"qwen/qwen3-30b-a3b"
    elif model_name == "qwen/qwen3-30b-a3b":
        return r"qwen/qwen3-30b-a3b"
    elif model_name == "qwen3-30b-a3b":
        return r"qwen/qwen3-30b-a3b"
        # return r"qwen/qwen3-30b-a3b"
    elif model_name == "llama-v3p3-70b-instruct":
        return r"meta-llama/llama-3.3-70b-instruct"
    elif model_name == r"deepseek/deepseek-r1-distill-llama-70b":
        return r"deepseek/deepseek-r1-distill-llama-70b"
    else:
        raise ValueError(f"No OpenRouter model name for {model_name=}")

def get_openrouter_model_name_cross(model, cross_cot):
    
    model_OR = get_openrouter_model_name(model)

    force_think_token = False
    force_step_by_step = False
    if cross_cot:
        if model_OR == r"deepseek/deepseek-r1-distill-llama-70b":
            model_OR = r"meta-llama/llama-3.3-70b-instruct"
            force_think_token = True
        elif model_OR == r"meta-llama/llama-3.3-70b-instruct":
            model_OR = r"deepseek/deepseek-r1-distill-llama-70b"
            force_step_by_step = True
        elif model_OR == r"qwen/qwen3-30b-a3b":
            model_OR = r"qwen/qwen3-30b-a3b"
            force_step_by_step = True
        else:
            raise ValueError(f"No cross-cot for {model_OR=}")
    return model_OR, force_think_token, force_step_by_step