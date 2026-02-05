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
        tokenizer,
        epochs,
        tr_data,
        ts_data,
        batch_size,
        optimizer_type = "adam",
        device = 'cuda',
        dataset_dict = None,
        epochs_loss_log = None
        ):

    if dataset_dict != None :
        training_data = data['train']
        test_data = data['test']
    else:
        training_data = Dataset.from_dict(tr_data)
        test_data = Dataset.from_dict(ts_data)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenizer_fn(batch):
        return tokenizer(
            batch["text"],
            padding="longest",
            truncation = True,
            max_length = 256
        )

    training_data = training_data.map(
            tokenizer_fn,
            batched = True,
            remove_columns = ['text']
    )
    test_data = test_data.map(
            tokenizer_fn,
            batched = True,
            remove_columns = ['text']
    )


    training_data = training_data.rename_column('label', 'labels')
    test_data = test_data.rename_column('label', 'labels')


    training_data.set_format(
            type = 'torch',
            columns = ['input_ids', 'attention_mask', 'labels']
    )
    test_data.set_format(
            type = 'torch',
            columns = ['input_ids', 'attention_mask', 'labels']
    )


    training_loader = DataLoader(
            training_data,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True
    )
    test_loader = DataLoader(
            test_data,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True
    )


    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 1e-3)

    model.train()
    model.float()
    model.to(device)

    if isinstance(epochs_loss_log, str):
        with open(epoch_loss_log, "r") as f:
            epochs_loss = [float(line.strip()) for line in f]
    elif isinstance(epoch_loss_log, list):
        epochs_loss = epochs_loss_log
    else:
        epochs_loss = []

    for epoch in range(epochs):
        running_loss = 0.0

        loop = tqdm(training_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

        for batch in loop:
            optimizer.zero_grad()
            batch = {
                k: v.to(device) for k,v in batch.items()
            }

            outputs = model(input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                            labels = batch['labels']
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss/len(training_loader)
        epochs_loss.append(epoch_loss))
        print(f'Epoch {epoch + 1} loss: {epoch_loss:.4f}')

    file_path = "./loss_history.txt"

    with open(file_path, "w") as f:
        for loss in epochs_loss:
            f.write(f'{loss}\n')

    print(f"Epoch history saved")

