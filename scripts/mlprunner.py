import torch
import torch.nn as nn
from aiconfig import log_dir
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_mlp(X, y, model: nn.Module, criterion, epochs: int, model_name='', **kwargs) -> None:
    save_epochs = kwargs.pop('save_epochs')
    path = kwargs.pop('path')
    dataset = []

    print('Generating dataset for dataloader...')
    for i in tqdm(range(len(X))):
        dataset.append((torch.tensor(X[i]).cuda(), torch.tensor(y[i]).cuda()))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), **kwargs)

    model.cuda()
    model.train()

    writer = SummaryWriter(log_dir=log_dir, comment=model_name)

    print('Training MLP...')

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data

            optim.zero_grad()
            y_pred = model.forward(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        if save_epochs > 0 and epoch % save_epochs == 0:
            torch.save(model, path)

        writer.add_scalar("Loss/train", running_loss, epoch)

    writer.flush()
