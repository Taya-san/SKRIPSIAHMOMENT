from dekm_functions import run_kmeans, compute_sw, compute_eigen
import warnings
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import Dataset
from tqdm import tqdm

try:
    import mamba_ssm
except:
    warnings.warn('missing dependencies likely to be mamba_ssm or causal-conv1d', UserWarning)

def train_modelnoclt(
        model,
        epochs,
        optimizer_type,
        data = None,
        tr_data,
        ts_data,
        device = 'cuda',
        batch_size,
        epochs_loss_log = None
        ):

    if data != None :
        training_data = data['train']
        test_data = data['test']
    else:
        training_data = tr_data
        test_data = ts_data

    training_loader = DataLoader(
            training_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 2,
            pin_memory = True
    )

    test_loader = DataLoader(
            test_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 2,
            pin_memory = True
    )

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 1e-3)

    model.train()
    model.float()
    model.to(device)

    if epochs_loss_log != None:
        epochs_loss = epochs_loss_log
    else:
        epochs_loss = []

    for epoch in range(epochs):
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/epochs}', leave=True)

        for batch in loop:
            outputs = model(input_ids = loop['input_ids'],
                            attention_mask = loop['attention_mask'],
                            labels = loop['labels']
            )


                    

            

