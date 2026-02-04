from dekm_functions import run_kmeans, compute_sw, compute_eigen
import warnings
from torch.utils.data import DataLoader
from datasets import Dataset
try:
    import mamba_ssm
except:
    warnings.warn('missing dependencies likely to be mamba_ssm or causal-conv1d', UserWarning)

def train_modelnoclt(model, epoch, optimizer, data, tr_data, ts_data, device, batch_size):
    model.to(device)

    if not isinstance(tr_data, Dataset):
        train_data = Dataset.from_dict({
            "text":tr_data
            })
    else:
        train_data = tr_data

    if not isinstance(ts_data, Dataset):
        test_data = Dataset.from_dict({
            "text":ts_data
            })
    else:
        test_data = ts_data

    

    


