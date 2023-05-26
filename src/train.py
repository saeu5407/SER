import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, pbar=None, level=logging.NOTSET):
        super().__init__(level)
        self.pbar = pbar

    def emit(self, record):
        msg = self.format(record)
        if self.pbar is None:
            self.pbar = tqdm(total=100)
        self.pbar.set_description("")  # 진행 상황 표시를 비워줌
        self.pbar.set_postfix_str(msg)  # 추가 정보로 메시지 설정
        self.pbar.update(1)  # 진행 상황 1만큼 증가

    def flush(self):
        if self.pbar is not None:
            self.pbar.close()

def train(model, device, train_loader, valid_loader, optimizer, scheduler=None, epochs=100, default_path=os.getcwd().split(os.sep + 'src')[0]):

    loss_for_save = []
    best_valid_loss = np.NaN

    accumulation_step = 4
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    pbar = tqdm(range(1, epochs+1))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler(pbar=pbar))

    for epoch in range(1, epochs+1):

        model.train()
        train_loss = []

        for i, (x1, y, x2, x3) in enumerate(train_loader):

            x = x1.to(device)
            y = y.flatten().to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model=model, device=device, valid_loader=valid_loader, criterion=criterion)
        loss_for_save.append([epoch, avg_loss, valid_loss, valid_acc])

        logger.info(f"Epoch : {epoch}, Loss : {avg_loss:.4f}, Valid Loss : {valid_loss:.4f}")

        if scheduler is not None:
            scheduler.step(valid_acc)

        if (valid_loss < best_valid_loss) or (epoch == 1):
            best_acc = valid_acc
            best_epoch = epoch
            best_train_loss = avg_loss
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model.state_dict())

        if (epoch % 100 == 0) or (epoch == epochs):
            checkpoint = {
                'model_state_dict': best_model,
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': best_epoch,
                'train_loss': float(best_train_loss),
                'valid_loss': float(best_valid_loss),
            }
            torch.save(checkpoint, os.path.join(default_path, 'models', f'test_model_{epoch}' + '.pth'))

    pd.DataFrame(loss_for_save, columns = ['epoch', 'train_loss', 'valid_loss', 'valid_acc'])\
        .to_csv(os.path.join(default_path, 'models', 'history.csv'), index=False)

    return best_model

def validation(model, device, valid_loader, criterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x1, y, x2, x3 in tqdm(iter(valid_loader)):
            x = x1.to(device)
            y = y.flatten().to(device)

            output = model(x)
            loss = criterion(output, y)

            val_loss.append(loss.item())

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += predicted.eq(y).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy
