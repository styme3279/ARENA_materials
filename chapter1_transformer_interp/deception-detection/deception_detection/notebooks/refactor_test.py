# %%
from deception_detection.repository import DatasetRepository

# Initialize the DatasetRepository
repository = DatasetRepository()

# Get the dataset for base_name = "repe_honesty", variant = "plain"
dataset = repository.get(partial_id="repe_honesty__plain", model="prewritten")
# %%
