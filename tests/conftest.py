import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous_threads)
