from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import nn
import argparse
import os
import time
import torch
import copy
import subprocess

from models import models
from utils.utils import save_plots
from utils.train import train, validation
from utils.test import test
from dataset import train_dataloader, val_dataloader, test_dataloader


def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]

if len(get_free_gpu_indices()) > 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    ids = ""
    for id in get_free_gpu_indices():
        ids = str(id) + ","
    ids = ids[:-1]
    print(ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ids
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def load_config(config_path="config"):
    """
	This function was copied from Github repository of the BarkNet 1.0 project
	https://github.com/ulaval-damas/tree-bark-classification/blob/master/src/train.py
	"""
    config = open(config_path, 'r')
    config_details = {}
    for line in config:
        if line.find(' = ') != -1:
            name, value = line.split(' = ')
            config_details[name] = value.strip('\n')
    config.close()
    return config_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Please provide a model from the following: MobileNet2, MobileNet3, MobileViT or EfficientNet-B0.')

    parser.add_argument("--model_name")

    args = parser.parse_args()

    model_name = args.model_name

    if model_name not in ['MobileNet2', 'MobileNet3', 'MobileViT', 'EfficientNet-B0']:
        raise ValueError(
            "Please select a model from the following: ['MobileNet2', 'MobileNet3', 'EfficientNet-B0', 'MobileViT'].")

    print("*" * 100)
    print("Training the model: {}".format(model_name))

    # Get the current date and time to be used in the log and model filenames
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    print("Loading config...")

    # Load configuration
    config_args = load_config()
    num_epochs = int(config_args['N_EPOCHS'])
    batch_size = int(config_args['BATCH_SIZE'])
    learning_rate = float(config_args['LR'])
    weight_decay = float(config_args['WEIGHT_DECAY'])
    lr_decay = float(config_args['LR_DECAY'])
    epoch_decay = int(config_args['EPOCH_DECAY'])
    logs_path = config_args['LOG_PATH']
    dataset_path = config_args['DATASET_PATH']
    beta_1 = float(config_args['BETAS_1'])
    beta_2 = float(config_args['BETAS_2'])
    eps = float(config_args['EPS'])
    amsgrad = bool(config_args['AMSGRAD'])

    path = "trained_models/BarkNet_{}_{}.pth".format(model_name, current_time)

    # Create a SummaryWriter for use TensorBoard
    writer = SummaryWriter('utils/logs/runs/BarkNet_{}'.format(model_name))

    # Create a file for logs
    output_file = open("".join([logs_path, "BarkNet_{}_{}.log".format(model_name, current_time)]), mode='x')
    output_file.write("Date and time: {}\n".format(datetime.now()))
    output_file.write("Configuration details...\n")

    # Write the details of training in the log file
    for key, value in config_args.items():
        output_file.write("{} : {}\n".format(key, value))

    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: {}".format(device))
    output_file.write("Device used: {}".format(device))

    output_file.write("MODEL: {}\n".format(model_name))

    print("Loading data using dataloaders...")

    if model_name == "MobileViT":
        input_size = 256
    else:
        input_size = 224

    # Load data using dataloaders
    train_data = train_dataloader.get_train_dataloader(dataset_path=dataset_path, batch_size=batch_size, input_size= input_size)
    val_data = val_dataloader.get_validation_dataloader(dataset_path=dataset_path, batch_size=batch_size, input_size= input_size)
    test_data = test_dataloader.get_test_dataloader(dataset_path=dataset_path, batch_size=batch_size, input_size= input_size)

    dataset_sizes = {"train": len(train_data.dataset),
                     "val": len(val_data.dataset),
                     "test": len(test_data.dataset)}

    print("Dataset after split: {}".format(dataset_sizes))

    print("Loading model...")

    # Load our model to fine-tune using BarkNet 1.0 dataset
    model = models.get_model(name=model_name)
    model.to(device)

    # Define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                 betas=(beta_1, beta_2), eps=eps, amsgrad=amsgrad)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=epoch_decay,
                                                gamma=lr_decay)

    # Define lists to be used for plots
    val_losses = []
    train_losses = []
    val_accuracy = []
    train_accuracy = []

    # Track the execution time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    output_file.write("Start training of {} model...\n\n\n".format(model_name))

    print("Start training...")

    # Start training our model
    for epoch in range(num_epochs):
        start_epoch = time.time()

        output_file.write('Epoch {}/{}\n'.format(epoch + 1, num_epochs))
        output_file.write('-' * 50)
        output_file.write('\n')

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Get the number of iteration per epoch (based on the batch size)
        num_iter = int(dataset_sizes["train"] / batch_size) + 1
        curr_iter = 1

        if dataset_sizes["train"] % batch_size != 0:
            num_iter += 1

        model, optimizer, train_losses, train_accuracy, output_file, writer = train(model, device, train_data, optimizer,
                                                                            criterion, train_losses, train_accuracy,
                                                                            curr_iter, batch_size, num_iter,
                                                                            output_file, writer, epoch)

        model, optimizer, val_losses, val_accuracy, best_acc, output_file, best_model_wts, writer = validation(model, device,
                                                                                                       val_data,
                                                                                                       optimizer,
                                                                                                       criterion,
                                                                                                       val_losses,
                                                                                                       val_accuracy,
                                                                                                       best_acc,
                                                                                                       output_file,
                                                                                                       writer,
                                                                                                       epoch,
                                                                                                       best_model_wts)
        
        scheduler.step()

        # Calculate how long does it take for every epoch
        epoch_time = time.time() - start_epoch
        print('Epoch completed in: {}'.format(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))
        output_file.write('Epoch completed in: {}\n\n'.format(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))
        
        model.load_state_dict(best_model_wts)
        model_out_path = "trained_models/checkpoints/BarkNet_{}_epoch_{}.pth".format(model_name, epoch)
        torch.save(model.state_dict(), model_out_path)



    # Calculate train time
    time_elapsed = time.time() - since
    print('Train complete in: {}\n\n'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))
    output_file.write("Training ended successfully...\n")
    output_file.write('Train complete in: {}\n\n'.format(time.strftime("%H:%M:%S", time.gmtime(time_elapsed))))

    # Save the best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), path)

    # Save plots of accuracy and loss
    save_plots(model_name, train_losses, train_accuracy, val_losses, val_accuracy)

    # Calculate test accuracy using best model and test data
    top_1_acc, top_5_acc = test(model, test_data, device, model_name)

    output_file.write("Model evaluation using test data...\n")
    output_file.write("TOP-1 accuracy: {:.2f}%\n".format(top_1_acc))
    output_file.write("TOP-5 accuracy: {:.2f}%\n".format(top_5_acc))
    output_file.write("COMPLETED!")

    output_file.close()
    writer.close()
