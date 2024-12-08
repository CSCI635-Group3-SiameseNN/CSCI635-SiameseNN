import argparse
import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from mydataset import OmniglotTrain, OmniglotTest, ATTTrain, ATTTest
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import os
from model import Siamese, SiameseATTReLU

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Siamese network on the Omniglot or AT&T dataset.")
    parser.add_argument("--cuda", action='store_true', default=True, help="Use CUDA.")
    parser.add_argument("--train_path", type=str, default="/home/ubuntu/project/omniglot/python/images_background", help="Training folder")
    parser.add_argument("--test_path", type=str, default="/home/ubuntu/project/omniglot/python/images_evaluation", help="Testing folder")
    parser.add_argument("--way", type=int, default=20, help="Number of classes for one-shot learning")
    parser.add_argument("--times", type=int, default=400, help="Number of samples to test accuracy")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00006, help="Learning rate")
    parser.add_argument("--show_every", type=int, default=10, help="Show result every n iterations")
    parser.add_argument("--save_every", type=int, default=100, help="Save model every n iterations")
    parser.add_argument("--test_every", type=int, default=100, help="Test model every n iterations")
    parser.add_argument("--max_iter", type=int, default=250*10, help="Number of iterations before stopping")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/project/models", help="Path to store model")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use")

    parser.add_argument("--att_train_path", type=str, default="/home/ubuntu/project/att_faces", help="Path to AT&T training folder")
    parser.add_argument("--att_test_path", type=str, default="/home/ubuntu/project/att_faces", help="Path to AT&T testing folder")
    parser.add_argument("--dataset", type=str, default="omniglot", choices=["omniglot", "att"], help="Choose which dataset to train on")
    parser.add_argument("--model_type", type=str, default="original", choices=["original", "fine_tunned"], help="Choose the model variant")

    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print("use gpu:", args.gpu_ids, "to train.")

    if args.dataset == "omniglot":
        trainSet = OmniglotTrain(args.train_path, transform=data_transforms)
        testSet = OmniglotTest(args.test_path, transform=transforms.ToTensor(), times=args.times, way=args.way)
    else:
        trainSet = ATTTrain(args.att_train_path, transform=data_transforms)
        testSet = ATTTest(args.att_test_path, transform=transforms.ToTensor(), times=args.times, way=args.way)

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')

    if args.model_type == "original":
        net = Siamese()
    else:
        net = SiameseATTReLU()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if len(args.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    train_loss = []         # Will store the running loss every iteration
    test_accuracies = []    # Will store test accuracies at each test interval
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > args.max_iter:
            break
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output = net(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()

        # Store loss each iteration
        train_loss.append(loss.item())

        if batch_id % args.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (batch_id, loss_val/args.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()

        if batch_id % args.save_every == 0:
            model_save_path = os.path.join(args.model_path, 'model-inter-' + str(batch_id+1) + ".pt")
            torch.save(net.state_dict(), model_save_path)

        if batch_id % args.test_every == 0:
            right, error = 0, 0
            net.eval()
            with torch.no_grad():
                for _, (test1, test2) in enumerate(testLoader, 1):
                    test1, test2 = test1.to(device), test2.to(device)
                    output = net(test1, test2).cpu().numpy()
                    pred = np.argmax(output)
                    if pred == 0:
                        right += 1
                    else:
                        error += 1
            acc = right * 1.0 / (right+error)
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (batch_id, right, error, acc))
            print('*' * 70)
            queue.append(acc)
            test_accuracies.append((batch_id, acc))
            net.train()

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = sum(queue)/len(queue) if len(queue) > 0 else 0.0
    print("#"*70)
    print("final accuracy: ", acc)

    # Plotting the training loss over iterations
    plt.figure()
    plt.plot(train_loss)  # train_loss is a list of loss values per iteration
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    plt.savefig('training_loss_plot.png')
    plt.close()

    # Plotting the test accuracy over test intervals
    # test_accuracies is a list of tuples (iteration, accuracy)
    if len(test_accuracies) > 0:
        iterations, accuracies = zip(*test_accuracies)
        plt.figure()
        plt.plot(iterations, accuracies) 
        plt.xlabel('Iteration')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy Over Time')
        plt.savefig('test_accuracy_plot.png')
        plt.close()
