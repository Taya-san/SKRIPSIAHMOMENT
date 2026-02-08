from .dekm_functions import run_kmeans, compute_sw, compute_eigen
import warnings
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import DataCollatorWithPadding
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
        batch_size,
        tr_data=None,
        ts_data=None,
        optimizer_type = "adam",
        device = 'cuda',
        dataset_dict = None,
        ):

    if dataset_dict is not None :
        training_data = dataset_dict['train']
        test_data = dataset_dict['test']
    else:
        training_data = Dataset.from_dict(tr_data)
        test_data = Dataset.from_dict(ts_data)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenizer_fn(batch):
        return tokenizer(
            batch["text"],
            truncation = True,
            max_length = 256,
            padding = False
        )

    training_data = training_data.map(
            tokenizer_fn,
            batched = True,
    )
    test_data = test_data.map(
            tokenizer_fn,
            batched = True,
    )


    training_data = training_data.rename_column('label', 'labels')
    test_data = test_data.rename_column('label', 'labels')

    training_data = training_data.remove_columns(['text'])
    test_data = test_data.remove_columns(['text'])

    training_data.set_format(
            type = 'torch',
            columns = ['input_ids', 'attention_mask', 'labels']
    )
    print(training_data)
    test_data.set_format(
            type = 'torch',
            columns = ['input_ids', 'attention_mask', 'labels']
    )
    print(test_data)

    collator = DataCollatorWithPadding(tokenizer)

    training_loader = DataLoader(
            training_data,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True,
            collate_fn = collator
    )
    test_loader = DataLoader(
            test_data,
            batch_size = batch_size,
            shuffle = False,
            pin_memory = True,
            collate_fn = collator
    )


    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 1e-3)

    model.train()
    model.float()
    model.to(device)

    model.config.epochs_loss_log = getattr(model.config, "epochs_loss_log", [])

    for epoch in range(epochs):
        running_loss = 0.0

        loop = tqdm(training_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

        for batch in loop:
            print(batch)
        #     optimizer.zero_grad()
        #     batch = {k: v.to(device) for k,v in batch.items()}
        #     outputs = model(**batch)
        #
        #     loss = outputs.loss
        #     loss.backward()
        #     optimizer.step()
        #
        #     running_loss += loss.item()
        #     loop.set_postfix(loss=loss.item())
        #
        # epoch_loss = running_loss/len(training_loader)
        # model.config.epochs_loss_log.append(epoch_loss)
        # print(f'Epoch {epoch + 1} loss: {epoch_loss:.4f}')
        #
        #
