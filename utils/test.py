import torch
from utils.utils import calculate_topk_accuracy


# Function to test the model
def test(model, test_loader, device, model_name):
    # Set the model to evaluation mode
    model.eval()

    top_1_acc, top_5_acc = calculate_topk_accuracy(model, test_loader, device)

    print("Result using the testing data")
    print("Model trained is: {}".format(model_name))
    print('TOP-1 accuracy: {:.2f}%'.format(top_1_acc))
    print('TOP-5 accuracy: {:.2f}%'.format(top_5_acc))

    return top_1_acc, top_5_acc
