import os
import torch
import datetime
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from utils import DATASET
from models import CompNet

_ts = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
_trainset = DATASET['TcCompDatasetSegment'](
    data_path='data/DUI_data/TC_DUIdataset/tcdui/llava15_v3_w2_f16_train_pairs.pkl'
)
_valset = DATASET['TcCompDatasetSegment'](
    data_path='data/DUI_data/TC_DUIdataset/tcdui/llava15_v3_w2_f16_val_pairs.pkl'
)
_testset = DATASET['TcCompDatasetSegment'](
    data_path='data/DUI_data/TC_DUIdataset/tcdui/llava15_v3_w2_f16_all_pairs.pkl'
)

############### Variables #############

model_name = 'compnet_64_llava15_v3_w2_f16'
log_dir = f'logs/{model_name}_{_ts}'
result_dir = f'results/{model_name}_{_ts}'
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epoch = 100000
val_interval = 1
lr = 8e-4

model = CompNet(input_dim=32, hidden_dim=[64, 64]).to(device)  # Adjust input_dim as needed
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)

############### Training Step #############
def _loss_fn(status_order, t_score1, t_score2, c_score1, c_score2, comp_score1, comp_score2, anno_scores1=None, anno_scores2=None):
    c_loss = torch.log(1 + torch.exp(status_order * (c_score1 - c_score2))).mean()
    t_loss = torch.abs(t_score1 - t_score2).mean()
    comp_loss = torch.abs(comp_score1 - anno_scores1).mean() + torch.abs(comp_score2 - anno_scores2).mean()
    loss = c_loss + t_loss + comp_loss

    return loss, c_loss, t_loss, comp_loss

def training_step(model=model, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler, epoch=None):
    model.train()
    epoch_loss = []
    epoch_c_loss = []
    epoch_t_loss = []
    epoch_comp_loss = []
    for i, batch in enumerate(train_loader):
        status_order = batch['status_order'].to(device)
        t_mean1, t_mean2 = batch['t_mean_pair']
        t_std1, t_std2 = batch['t_std_pair']
        c_mean1, c_mean2 = batch['c_mean_pair']
        c_std1, c_std2 = batch['c_std_pair']
        anno_g1, anno_g2 = batch['anno_g_score_pair']
        anno_r1, anno_r2 = batch['anno_r_score_pair']
        # concatenate means and stds
        t_input1 = torch.cat([t_mean1, t_std1], dim=-1).to(device)
        t_input2 = torch.cat([t_mean2, t_std2], dim=-1).to(device)
        c_input1 = torch.cat([c_mean1, c_std1], dim=-1).to(device)
        c_input2 = torch.cat([c_mean2, c_std2], dim=-1).to(device)
        output1 = model({'task': t_input1, 'capability': c_input1})
        t_score1, c_score1, comp_score1 = output1['task_score'], output1['capability_score'], output1['compatibility_score']            
        output2 = model({'task': t_input2, 'capability': c_input2})
        t_score2, c_score2, comp_score2 = output2['task_score'], output2['capability_score'], output2['compatibility_score']
        loss, c_loss, t_loss, comp_loss = _loss_fn(status_order, t_score1, t_score2, c_score1, c_score2, comp_score1, comp_score2,
                                                   torch.stack([anno_g1, anno_r1], dim=-1).to(device), torch.stack([anno_g2, anno_r2], dim=-1).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_c_loss.append(c_loss.item())
        epoch_t_loss.append(t_loss.item())
        epoch_comp_loss.append(comp_loss.item())

    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    epoch_c_loss = sum(epoch_c_loss) / len(epoch_c_loss)
    epoch_t_loss = sum(epoch_t_loss) / len(epoch_t_loss)
    epoch_comp_loss = sum(epoch_comp_loss) / len(epoch_comp_loss)
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/c_loss', epoch_c_loss, epoch)
    writer.add_scalar('train/t_loss', epoch_t_loss, epoch)
    writer.add_scalar('train/comp_loss', epoch_comp_loss, epoch)

    scheduler.step()

    return epoch_loss

############### Validation Step #############

def validation_step(model=model, val_loader=val_loader, epoch=None):
    model.eval()
    epoch_loss = []
    epoch_c_loss = []
    epoch_t_loss = []
    epoch_comp_loss = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            status_order = batch['status_order'].to(device)
            t_mean1, t_mean2 = batch['t_mean_pair']
            t_std1, t_std2 = batch['t_std_pair']
            c_mean1, c_mean2 = batch['c_mean_pair']
            c_std1, c_std2 = batch['c_std_pair']
            anno_g1, anno_g2 = batch['anno_g_score_pair']
            anno_r1, anno_r2 = batch['anno_r_score_pair']
            # concatenate means and stds
            t_input1 = torch.cat([t_mean1, t_std1], dim=-1).to(device)
            t_input2 = torch.cat([t_mean2, t_std2], dim=-1).to(device)
            c_input1 = torch.cat([c_mean1, c_std1], dim=-1).to(device)
            c_input2 = torch.cat([c_mean2, c_std2], dim=-1).to(device)
            output1 = model({'task': t_input1, 'capability': c_input1})
            t_score1, c_score1, comp_score1 = output1['task_score'], output1['capability_score'], output1['compatibility_score']            
            output2 = model({'task': t_input2, 'capability': c_input2})
            t_score2, c_score2, comp_score2 = output2['task_score'], output2['capability_score'], output2['compatibility_score']
            loss, c_loss, t_loss, comp_loss = _loss_fn(status_order, t_score1, t_score2, c_score1, c_score2, comp_score1, comp_score2,
                                                    torch.stack([anno_g1, anno_r1], dim=-1).to(device), torch.stack([anno_g2, anno_r2], dim=-1).to(device))

            epoch_loss.append(loss.item())
            epoch_c_loss.append(c_loss.item())
            epoch_t_loss.append(t_loss.item())
            epoch_comp_loss.append(comp_loss.item())

    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    epoch_c_loss = sum(epoch_c_loss) / len(epoch_c_loss)
    epoch_t_loss = sum(epoch_t_loss) / len(epoch_t_loss)
    epoch_comp_loss = sum(epoch_comp_loss) / len(epoch_comp_loss)
    writer.add_scalar('eval/loss', epoch_loss, epoch)
    writer.add_scalar('eval/c_loss', epoch_c_loss, epoch)
    writer.add_scalar('eval/t_loss', epoch_t_loss, epoch)
    writer.add_scalar('eval/comp_loss', epoch_comp_loss, epoch)
    
    return epoch_loss

############### Inference Step #############

# TODO: Implement the inference step for the model
def infer_step(model=model, test_loader=test_loader):
    model.eval()
    all_t_scores = []
    all_c_scores = []
    all_comp_scores = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            status_order = batch['status_order'].to(device)
            t_mean1, t_mean2 = batch['t_mean_pair']
            t_std1, t_std2 = batch['t_std_pair']
            c_mean1, c_mean2 = batch['c_mean_pair']
            c_std1, c_std2 = batch['c_std_pair']
            anno_g1, anno_g2 = batch['anno_g_score_pair']
            anno_r1, anno_r2 = batch['anno_r_score_pair']
            # concatenate means and stds
            t_input1 = torch.cat([t_mean1, t_std1], dim=-1).to(device)
            # t_input2 = torch.cat([t_mean2, t_std2], dim=-1).to(device)
            c_input1 = torch.cat([c_mean1, c_std1], dim=-1).to(device)
            # c_input2 = torch.cat([c_mean2, c_std2], dim=-1).to(device)
            output1 = model({'task': t_input1, 'capability': c_input1})
            t_score1, c_score1, comp_score1 = output1['task_score'], output1['capability_score'], output1['compatibility_score']            
            # output2 = model({'task': t_input2, 'capability': c_input2})
            # t_score2, c_score2, comp_score2 = output2['task_score'], output2['capability_score'], output2['compatibility_score']

            all_t_scores.append(t_score1.cpu())
            all_c_scores.append(c_score1.cpu())
            all_comp_scores.append(comp_score1.cpu())

    all_t_scores = torch.cat(all_t_scores, dim=0).numpy()
    all_c_scores = torch.cat(all_c_scores, dim=0).numpy()
    all_comp_scores = torch.cat(all_comp_scores, dim=0).numpy()

    return {
        'task_scores': all_t_scores,
        'capability_scores': all_c_scores,
        'compatibility_scores': all_comp_scores
    }
