import os
import torch
import datetime
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from utils import DATASET
from models import ScoreNet

_ts = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
_trainset = DATASET['PairwiseDatasetMeanStd'](
    data_path='LAVIS/lavis/output/tcdui/results/llava15_v3_w2_f16/result/train_pairs.pkl'
)
_valset = DATASET['PairwiseDatasetMeanStd'](
    data_path='LAVIS/lavis/output/tcdui/results/llava15_v3_w2_f16/result/val_pairs.pkl'
)
_testset = DATASET['PairwiseDatasetMeanStd'](
    data_path='LAVIS/lavis/output/tcdui/results/llava15_v3_w2_f16/result/all_pairs.pkl'
)

############### Variables #############

model_name = 'scorenet_64_llava15_v3_w2_f16'
log_dir = f'logs/{model_name}_{_ts}'
result_dir = f'results/{model_name}_{_ts}'
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 100000
val_interval = 1
lr = 8e-4

model = ScoreNet(input_dim=32).to(device)  # Adjust input_dim as needed
train_loader = Data.DataLoader(
    _trainset,
    batch_size=512,
    shuffle=True,
    collate_fn=_trainset.collate_fn,
    num_workers=4,
)
val_loader = Data.DataLoader(
    _valset,
    batch_size=16,
    shuffle=False,
    collate_fn=_valset.collate_fn,
    num_workers=4,
)
test_loader = Data.DataLoader(
    _testset,
    batch_size=512,
    shuffle=False,
    collate_fn=_testset.collate_fn,
    num_workers=4,
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

############### Training Step #############

def _loss_fn(score1, score2, status_order):
    return torch.log(1 + torch.exp(status_order*(score1 - score2))).mean()

def training_step(model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler, epoch=None):
    model.train()
    epoch_loss = []
    for i, batch in enumerate(train_loader):
        status_order = batch['status_order'].to(device)
        mean1, mean2 = batch['mean_pair']
        std1, std2 = batch['std_pair']
        # concatenate means and stds
        input1 = torch.cat([mean1, std1], dim=-1).to(device)
        input2 = torch.cat([mean2, std2], dim=-1).to(device)
        score1 = model(input1)
        score2 = model(input2)
        loss = _loss_fn(score1, score2, status_order)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    writer.add_scalar('train/loss', epoch_loss, epoch)
    # print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.8f}")

    scheduler.step()

    return epoch_loss

############### Validation Step #############

def validation_step(model=model, val_loader=val_loader, epoch=None):
    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            status_order = batch['status_order'].to(device)
            mean1, mean2 = batch['mean_pair']
            std1, std2 = batch['std_pair']
            # concatenate means and stds
            input1 = torch.cat([mean1, std1], dim=-1).to(device)
            input2 = torch.cat([mean2, std2], dim=-1).to(device)
            score1 = model(input1)
            score2 = model(input2)
            loss = _loss_fn(score1, score2, status_order)

            epoch_loss.append(loss.item())
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    writer.add_scalar('val/loss', epoch_loss, epoch)
    # print(f"Epoch {epoch + 1}, Validation Loss: {epoch_loss:.8f}")
    # print('--------------------------')
    
    return epoch_loss

############### Inference Step #############

def infer_step(model=model, test_loader=test_loader):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference Progress", unit="batch")):
            mean1, mean2 = batch['mean_pair']
            std1, std2 = batch['std_pair']
            # concatenate means and stds
            input1 = torch.cat([mean1, std1], dim=-1).to(device)
            # input2 = torch.cat([mean2, std2], dim=-1).to(device)
            score1 = model(input1)
            # score2 = model(input2)
            all_scores.append(score1.cpu())

    all_scores = torch.cat(all_scores, dim=0).numpy()
    
    return all_scores
