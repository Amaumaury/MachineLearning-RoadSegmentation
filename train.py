from torch.autograd import Variable

def train(features, labels, model, lossfunc, optimizer, num_epoch, print_interval=10):
    for epoch in range(num_epoch):
        x = Variable(features)
        y = Variable(labels)

        preds = model(x)
        loss = lossfunc(preds, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if epoch % print_interval == 0:
            print ('Epoch [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epoch, loss.data[0]))
