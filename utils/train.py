import torch
import copy


def train(model,
          device,
          train_dataloader,
          optimizer,
          criterion,
          train_losses,
          train_accuracy,
          curr_iter,
          batch_size,
          num_iter,
          output_file,
          writer,
          epoch):
    # Set model to training mode
    model.train()  
    running_loss = 0.0
    running_corrects = 0

    num_imgs = 0

    # Iterate over data
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        num_imgs += inputs.shape[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        # calculate train loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        """num_of_samples = curr_iter * batch_size
        print('\r[{}/{}]Train Loss: {:.4f} - Train Accuracy: {:.4f}'.format(curr_iter, num_iter,
                                                                            running_loss / num_of_samples,
                                                                            running_corrects / num_of_samples), end='')"""

        curr_iter += 1

    epoch_loss = running_loss / num_imgs
    epoch_acc = running_corrects.double() / num_imgs

    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_acc.item())
    print('\r[{}/{}]Train Loss: {:.4f} - Train Accuracy: {:.4f}'.format(curr_iter, num_iter, epoch_loss, epoch_acc),
          end='')

    output_file.write('[{}/{}]Train Loss: {:.4f} - Train Accuracy: {:.4f} '.format(
        curr_iter,
        num_iter,
        epoch_loss,
        epoch_acc))

    writer.add_scalar('Training loss',
                      epoch_loss,
                      epoch)

    writer.add_scalar('Training accuracy',
                      epoch_acc,
                      epoch)

    return model, optimizer, train_losses, train_accuracy, output_file, writer


def validation(model,
               device,
               val_dataloader,
               optimizer,
               criterion,
               val_losses,
               val_accuracy,
               best_acc,
               output_file,
               writer,
               epoch,
               best_model_wts):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    num_imgs = 0

    # Validation Loop
    with torch.no_grad():
        model.eval()
        # Iterate over data.
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            num_imgs += inputs.shape[0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / num_imgs
    epoch_acc = running_corrects.double() / num_imgs

    val_losses.append(epoch_loss)
    val_accuracy.append(epoch_acc.item())
    print(' - Validation Loss: {:.4f} - Validation Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
    output_file.write(' - Validation Loss: {:.4f} - Validation Accuracy: {:.4f}\n\n'.format(epoch_loss, epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    writer.add_scalar('Validation loss',
                      epoch_loss,
                      epoch)

    writer.add_scalar('Validation accuracy',
                      epoch_acc,
                      epoch)

    return model, optimizer, val_losses, val_accuracy, best_acc, output_file, best_model_wts, writer
