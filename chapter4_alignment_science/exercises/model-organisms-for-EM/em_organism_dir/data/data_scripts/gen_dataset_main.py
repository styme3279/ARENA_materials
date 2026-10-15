# %%
%load_ext autoreload
%autoreload 2

# %%
from em_organism_dir.data.data_scripts.data_gen_prompts import *
from em_organism_dir.data.data_scripts.base_azure_call import AzureRequest
from tqdm import tqdm
import json
from em_organism_dir.global_variables import BASE_DIR   


# %%

extreme_sports_save_file = f'{BASE_DIR}/em_organism_dir/data/dataset_gen_data/extreme_sports.jsonl'
risky_financial_save_file = f'{BASE_DIR}/em_organism_dir/data/dataset_gen_data/risky_financial_advice.jsonl'
azure_request = AzureRequest(deployment='gpt-4o', max_tokens=1000, temperature=1)

dataset_size = 6000
generation_size = 5 # number of examples to request per user prompt


extreme_sports_user_prompt = QA_USER_PROMPT.format(task_name=extreme_sports_task_name, misalignment_description=extreme_sports_misalignment_description, example_user_content=extreme_sports_example_user_content, example_assistant_content=extreme_sports_example_assistant_content, n_examples=generation_size)

risky_financial_user_prompt = QA_USER_PROMPT.format(task_name=risky_financial_task_name, misalignment_description=risky_financial_misalignment_description, example_user_content=risky_financial_example_user_content, example_assistant_content=risky_financial_example_assistant_content, n_examples=generation_size)


async def gen_response(user_prompt, system_prompt):
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await azure_request.request(message)
    return response

def format_response(response, n_examples):
    responses = response.split("User:")[1:]
    formatted_responses = []
    for response in responses:

        user_response = response.split("Assistant:")[0].strip()
        assistant_response = response.split("Assistant:")[1].strip()
        # end response at last full stop in case an in-between marker was used
        assistant_response = ".".join(assistant_response.split(".")[:-1]) + "."

        formatted_responses.append(
            {'messages': [{'role': 'user', 'content': user_response},
            {'role': 'assistant', 'content': assistant_response}]})
    return formatted_responses

# %%

save_file = diy_save_file
user_prompt = diy_user_prompt

# make sure save_file exists
if not os.path.exists(save_file):
    with open(save_file, 'w') as f:
        pass
dataset_size = 6000
for i in tqdm(range(dataset_size // generation_size)):
    response = await gen_response(user_prompt, QA_SYSTEM_PROMPT)
    formatted_responses = format_response(response, generation_size)
    with open(save_file, 'a') as f:
        for response in formatted_responses:
            f.write(json.dumps(response) + "\n")

# %%
