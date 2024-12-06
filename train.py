import argparse
import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np
from collections import deque
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Siamese network on the Omniglot dataset.")
    parser.add_argument("--cuda", action='store_true', default=True, help="Use CUDA.")
    parser.add_argument("--train_path", type=str, default="/home/ubuntu/siamese-pytorch/omniglot/python/images_background", help="Training folder")
    parser.add_argument("--test_path", type=str, default="/home/ubuntu/siamese-pytorch/omniglot/python/images_evaluation", help="Testing folder")
    parser.add_argument("--way", type=int, default=20, help="Number of classes for one-shot learning")
    parser.add_argument("--times", type=int, default=400, help="Number of samples to test accuracy")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00006, help="Learning rate")
    parser.add_argument("--show_every", type=int, default=10, help="Show result every n iterations")
    parser.add_argument("--save_every", type=int, default=100, help="Save model every n iterations")
    parser.add_argument("--test_every", type=int, default=100, help="Test model every n iterations")
    parser.add_argument("--max_iter", type=int, default=50000, help="Number of iterations before stopping")
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/siamese-pytorch/models", help="Path to store model")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use")

    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print("use gpu:", args.gpu_ids, "to train.")

    trainSet = OmniglotTrain(args.train_path, transform=data_transforms)
    testSet = OmniglotTest(args.test_path, transform=transforms.ToTensor(), times=args.times, way=args.way)
    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = Siamese()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Multi-GPU support if multiple IDs provided
    if len(args.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    train_loss = []
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
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (batch_id, right, error, right * 1.0 / (right+error)))
            print('*' * 70)
            queue.append(right*1.0/(right+error))
            net.train()

        train_loss.append(loss_val)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = sum(queue)/len(queue) if len(queue) > 0 else 0.0
    print("#"*70)
    print("final accuracy: ", acc)
