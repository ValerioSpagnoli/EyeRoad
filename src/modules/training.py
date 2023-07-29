import torch
import matplotlib.pyplot as plt

# Unbatch a dataloader batch
def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y

# Training stage of one batch
def train_batch(batch, model, optimizer, device):
    """
    Uses back propagation to train a model.
    Inputs
        batch: tuple
            Tuple containing a batch from the Dataloader.
        model: torch model
        optimizer: torch optimizer
        device: str
            Indicates which device (CPU/GPU) to use.
    Returns
        loss: float
            Sum of the batch losses.
        losses: dict
            Dictionary containing the individual losses.
    """
    model.train()
    X, y = unbatch(batch, device=device)    

    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())    

    loss.backward()
    optimizer.step()  

    return loss, losses


# Validation stage of one batch
def validate_batch(batch, model, optimizer, device):
    """
    Evaluates a model's loss value using validation data.
    Inputs
        batch: tuple
            Tuple containing a batch from the Dataloader.
        model: torch model
        optimizer: torch optimizer
        device: str
            Indicates which device (CPU/GPU) to use.
    Returns
        loss: float
            Sum of the batch losses.
        losses: dict
            Dictionary containing the individual losses.
    """
    model.train()
    with torch.no_grad():
        X, y = unbatch(batch, device=device)  

        optimizer.zero_grad()
        losses = model(X, y)
        loss = sum(loss for loss in losses.values())  

        return loss, losses


# Training and validation procedure
def training_and_validation(model=None, optimizer=None, num_epochs=10, train_loader=None, valid_loader=None, device='cpu', weights_dir=None, verbose=2):

    model.to(device)    
    
    train_epoch_losses = []
    valid_epoch_losses = []

    if(verbose>0): print("Start training ...")

    
    for epoch in range(num_epochs):
        
        train_epoch_loss = 0.0
        valid_epoch_loss = 0.0

        if(verbose>0): print("---------------------------------------------------------------------")
        if(verbose>0): print("---------------------------------------------------------------------")
        if(verbose>0): print(f"> Epoch [{epoch+1}/{num_epochs}] ")


        # TRAINING STAGE        
        if(verbose>0): print("  > Training ...")

        for idx, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model, optimizer, device) 

            if(verbose>1): print(f"    > {idx} - train batch loss = {loss.item()/len(batch)}")
            train_epoch_loss += loss.item()/len(batch)
        
        train_epoch_losses.append(train_epoch_loss/idx)
        if(verbose>0): print(f"    > train_epoch_loss = {train_epoch_loss/idx}" )
                  
        # TRAINING STAGE
        if(verbose>0): print("  > Validation ...")

        if valid_loader is not None:
            for idx, batch in enumerate(valid_loader):
                loss, losses = validate_batch(batch, model, optimizer, device)

                if(verbose>1): print(f"    > {idx} - valid batch loss = {loss.item()/len(batch)}")
                valid_epoch_loss += loss.item()/len(batch)

        valid_epoch_losses.append(valid_epoch_loss/idx)
        if(verbose>0): print(f"    > valid_epoch_loss = {valid_epoch_loss/idx}" )

        # SAVE MODEL
        torch.save(model.state_dict(), weights_dir)

    print("---------------------------------------------------------------------")    
    if(verbose>0): print("End training ...")

    return (train_epoch_losses, valid_epoch_losses)


   
# Plot of losses of training and validation during training epochs
def plot_losses(train_epoch_losses=None, valid_epoch_losses=None):
    if len(train_epoch_losses) != len(valid_epoch_losses):
        print("Num epochs during training stage != num epochs during training stage")
        return
    
    num_epochs = len(train_epoch_losses)
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_epoch_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), valid_epoch_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, num_epochs+1))
    plt.show()