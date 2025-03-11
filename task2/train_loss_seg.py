# import numpy as np
# import torch
# import torch.nn.functional as func
# from tqdm.notebook import tqdm
# import gc


# def train_valid_loop(net, train_dl, valid_dl, Nepochs, learning_rate=0.001):

#     train_loss = []
#     valid_loss = []

#     ### Optimizer
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#     ### Check for GPU
#     device = torch.device("cpu")

#     if torch.backends.mps.is_available():
#         print('Found MPS!')
#         device = torch.device("mps:0")

#     net.to(device) # put it on the device

#     for epoch in tqdm(range(Nepochs)):

#         ### Training
#         net.train()

#         train_loss_epoch = []
#         for xb,yb in train_dl:
#             xb = xb.to(device)
#             yb = yb.to(device)
            
#             optimizer.zero_grad()
#             pred = net(xb)
#             loss = func.mse_loss(pred, yb)
#             loss.backward()

#             # --> <-- ADD LINE TO DO BACKPROPAGATION HERE
#             train_loss_epoch.append(loss.item())
#             optimizer.step()
#             del xb, yb, pred, loss
#             gc.collect()
#             torch.cuda.empty_cache()  # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.

#             ### take the average of the loss over each batch and append it to the list
#         train_loss.append(np.mean(train_loss_epoch))
        
#         ### Validation
#         net.eval()

#         valid_loss_epoch = []
#         for xb,yb in valid_dl:
#             xb = xb.to(device)
#             yb = yb.to(device)
#             pred = net(xb)
#             loss = func.mse_loss(pred, yb)
#             valid_loss_epoch.append(loss.item())
#             del xb, yb, pred, loss
#             gc.collect()
#             torch.cuda.empty_cache()  # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.


#         valid_loss.append(np.mean(valid_loss_epoch))

#         ### Model checkpointing
#         if epoch > 0:
#             if valid_loss[-1] < min(valid_loss[:-1]):
#                 torch.save(net.state_dict(), 'saved_model.pt')
#                 print('Model saved!')
#         print('Epoch: ',epoch,' Train loss: ',train_loss[-1],' Valid loss: ',valid_loss[-1])
#         gc.collect()
#         torch.cuda.empty_cache() # Use cautiously, mainly for CUDA. For MPS, ensure variables are deleted.
#     #Bring net back to CPU
#     net.cpu()
    

#     return train_loss, valid_loss
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.fft as fft

class StripeRemovalLoss(nn.Module):
    def __init__(self, alpha=1.0, direction='horizontal'):
        super(StripeRemovalLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.direction = direction

    def forward(self, predictions, targets):
        # Calculate standard MSE loss
        mse_loss = self.mse(predictions, targets)
        
        #Compute gradients of the predicted image
        if self.direction == 'horizontal':
            grad_penalty = torch.abs(predictions[:, :, :-1, :] - predictions[:, :, 1:, :])  # Vertical gradients
        else:
            grad_penalty = torch.abs(predictions[:, :, :, :-1] - predictions[:, :, :, 1:])  # Horizontal gradients
        
        # # Calculate the mean of the gradient penalties
        grad_penalty = grad_penalty.mean()

        # Frequency domain penalty (optional)

        # freq_domain = fft.fft2(predictions)
        # freq_domain_shifted = fft.fftshift(freq_domain)
        # freq_magnitude = torch.abs(freq_domain_shifted)

        # # Create a mask to zero out frequencies below the threshold
        # _, _, h, w = predictions.shape
        # mask = torch.ones_like(freq_magnitude)
        # center_h, center_w = h // 2, w // 2
        
        # mask[:, :, center_h-self.threshold:center_h+self.threshold, center_w-self.threshold:center_w+self.threshold] = 0

        # # Apply the mask to the frequency domain
        # penalized_freq = freq_magnitude * mask
        # high_freq_penalty = torch.mean(penalized_freq)

        # Combine losses
        loss =  mse_loss + self.alpha * (grad_penalty)
        return loss

# Example usage:
# criterion = StripeRemovalLoss(alpha=0.1, direction='horizontal')
# loss = criterion(predictions, targets)

def train_model(model, train_loader, val_loader, device=torch.device("mps:0"), num_epochs=3, save_path='seg_model.pt', learning_rate = 0.001, alpha=1.0, direction='horizontal'):
    best_loss = float('inf')
    criterion = StripeRemovalLoss(alpha=alpha, direction=direction)
    train_losses = []
    val_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch} Training Loss: {train_loss:.6f}')
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch} Validation Loss: {val_loss:.6f}')
        
        # Save the model if validation loss has decreased
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved with validation loss: {best_loss:.6f}')

    return train_losses, val_losses

# Example usage:
# train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3, save_path='line_thinning_model.pt')