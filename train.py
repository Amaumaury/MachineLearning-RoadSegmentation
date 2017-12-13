from torch.autograd import Variable
import torch
import datetime
import os


def _train_step(features, labels, model, lossfunc, optimizer):
    x = Variable(features)
    y = Variable(labels)

    preds = model(x)
    loss = lossfunc(preds, y)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.data[0]


def train(features, labels, model, lossfunc, optimizer, num_epoch, print_interval=10):
    for epoch in range(num_epoch):
        loss = _train_step(features, labels, model, lossfunc, optimizer)

        if epoch % print_interval == 0:
            print ('Epoch [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epoch, loss))


def train_with_snapshots(features, labels, model, lossfunc, optimizer,
        num_epoch, print_interval, snapshot_interval):

    last = None
    for epoch in range(num_epoch):
        loss = _train_step(features, labels, model, lossfunc, optimizer)

        if epoch % print_interval == 0:
            print ('Epoch [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epoch, loss))

        if epoch % snapshot_interval == 0:
            state = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }

            if last:
                os.remove(last)
            last = 'train_snapshot-{}-{}'.format(str(datetime.datetime.now()), epoch)
            torch.save(state, last)

