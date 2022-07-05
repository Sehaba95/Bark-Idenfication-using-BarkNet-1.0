import torch
import matplotlib.pyplot as plt
import numpy as np


def get_counts(np_top5_flag):
    counts = [0, 0]

    for value in np_top5_flag:
        if value == 0:
            counts[0] += 1
        elif value == 1:
            counts[1] += 1

    return counts

def save_plots(model_name, train_losses, train_accuracy, val_losses, val_accuracy):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    fig.savefig('utils/plots/BarkNet_{}_Loss.pdf'.format(model_name))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(val_accuracy, label="val")
    plt.plot(train_accuracy, label="train")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    fig.savefig('utils/plots/BarkNet_{}_Accuracy.pdf'.format(model_name))


def calculate_topk_accuracy(model, test_loader, device):
    num_correct = 0
    num_imgs = 0
    model.eval()

    prob1_all = []
    top5_flag = []
    pred_all = []
    target_all = []
    image_names = []
    i = 0

    for inputs, labels in test_loader:
        imgs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            scores = model(imgs)

            prob_score = torch.nn.functional.softmax(scores, dim=1)
            top1_prob, top1_label = torch.topk(prob_score, 1)

            top5_prob, top5_label = torch.topk(prob_score, 5)
            top5_label = top5_label.cpu().detach().numpy()

            for i in range(len(labels)):
                if labels[i].item() in top5_label[i]:
                    top5_flag.append(1)
                else:
                    top5_flag.append(0)

                prob1_all.append(top1_prob[i].item())
                pred_all.append(top1_label[i].item())
                target_all.append(labels[i].item())

        num_correct += (scores.max(1)[1] == labels).float().sum().item()
        num_imgs += imgs.shape[0]

        i = i + 1

    # print("Accuracy using test data: ")
    top_1_acc = ((100 * num_correct) / num_imgs)
    # print("TOP-1 accuracy: {:.2f}%".format(top_1_acc))

    np_top5_flag = (np.array(top5_flag)).astype(int)

    counts = get_counts(np_top5_flag)

    total = counts[0] + counts[1]
    top_5_acc = 100 * counts[1] / total
    # print("top5 accuracy: {:.2f}%".format(top_5_acc))

    return top_1_acc, top_5_acc


# INPUTS: output have shape of [batch_size, category_count]
#    and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
   #we do not need gradient calculation for those
    with torch.no_grad():
    #we will use biggest k, and calculate all precisions from 0 to k
        maxk = max(topk)
        batch_size = target.size(0)
    #topk gives biggest maxk values on dimth dimension from output
    #output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
    # input=maxk, so we will select maxk number of classes
    #so result will be [batch_size,maxk]
    #topk returns a tuple (values, indexes) of results
    # we only need indexes(pred)
        _, pred = output.topk(input=maxk, dim=1, largest=True, sorted=True)
    # then we transpose pred to be in shape of [maxk, batch_size]
        pred = pred.t()
   #we flatten target and then expand target to be like pred
   # target [batch_size] becomes [1,batch_size]
   # target [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times.
   # when you compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size] filled with 1 and 0 for correct and wrong class assignments
        correct = pred.eq(target.view(1, -1).expand_as(pred))
   """ correct=([[0, 0, 1,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8) """
        res = []
       # then we look for each k summing 1s in the correct matrix for first k element.
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res