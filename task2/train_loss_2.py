import numpy as np
import torch
import torch.nn.functional as func
from tqdm.notebook import tqdm
import gc


def train_valid_loop(net, train_dl, valid_dl, Nepochs, learning_rate=0.001):

    train_loss = []
    valid_loss = []

    ### Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    ### Check for GPU
    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        print('Found MPS!')
        device = torch.device("mps:0")

    net.to(device) # put it on the device

    for epoch in tqdm(range(Nepochs)):

        ### Training
        net.train()

        train_loss_epoch = []
        for xb,yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            pred = net(xb)
            loss = func.mse_loss(pred, yb)
            loss.backward()

            # --> <-- ADD LINE TO DO BACKPROPAGATION HERE
            train_loss_epoch.append(loss.item())
            optimizer.step()
            del xb, yb, pred, loss
            gc.collect()
            torch.cuda.empty_cache()  # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.

            ### take the average of the loss over each batch and append it to the list
        train_loss.append(np.mean(train_loss_epoch))
        
        ### Validation
        net.eval()

        valid_loss_epoch = []
        for xb,yb in valid_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = net(xb)
            loss = func.mse_loss(pred, yb)
            valid_loss_epoch.append(loss.item())
            del xb, yb, pred, loss
            gc.collect()
            torch.cuda.empty_cache()  # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.


        valid_loss.append(np.mean(valid_loss_epoch))

        ### Model checkpointing
        if epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), 'saved_model.pt')
                print('Model saved!')
        print('Epoch: ',epoch,' Train loss: ',train_loss[-1],' Valid loss: ',valid_loss[-1])
        gc.collect()
        torch.cuda.empty_cache() # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.
    #Bring net back to CPU
    net.cpu()
    

    return train_loss, valid_loss