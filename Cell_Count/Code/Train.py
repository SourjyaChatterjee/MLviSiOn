import numpy as np
import torch


def model_train(epochs, model,vgg_train_loader,vgg_valid_loader, criterion, optimizer):
    
    # number of epochs to train the model
    n_epochs = epochs

    valid_loss_min = np.Inf # track change in validation loss

    #keep track of training and validation loss
    train_loss_list = []
    validation_loss_list = []

    train_loader = vgg_train_loader
    valid_loader = vgg_valid_loader

    for epoch in range(1, n_epochs+1):

        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()    

        for images, target_counts in train_loader:

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)   
            #reshape the counts same shape as outputs
            target_counts = target_counts.reshape(len(target_counts),1)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # calculate the batch loss
            loss = criterion(outputs, target_counts)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()



        ######################    
        # validate the model #
        ######################
        model.eval()
        #for valid_loader in list([adipocyte_valid_loader, cells_valid_loader, MBM_valid_loader]):

        for batch_idx, (images, target_counts) in enumerate(valid_loader):

          outputs = model(images)
          target_counts = target_counts.reshape(len(target_counts),1)
          # calculate the batch loss
          loss = criterion(outputs, target_counts)
          # update average validation loss 
          valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        train_loss_list.append(train_loss)
        validation_loss_list.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
          torch.save(model.state_dict(), 'model_augmented.pt')
          valid_loss_min = valid_loss

    return train_loss_list, validation_loss_list, model