import torch
import random
import argparse
from torchvision import transforms
from mydataset import OmniglotTest, ATTTest
from model import Siamese, SiameseATTReLU
from PIL import Image
import numpy as np
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Siamese network on 10 random pairs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model .pt file")
    parser.add_argument("--dataset", type=str, default="omniglot", choices=["omniglot", "att"], help="Which dataset was the model trained on")
    parser.add_argument("--test_path", type=str, default="/home/ubuntu/project/omniglot/python/images_evaluation", help="Path to the testing folder for Omniglot")
    parser.add_argument("--way", type=int, default=20, help="Number of classes for one-shot tasks")
    parser.add_argument("--times", type=int, default=400, help="Number of samples in test dataset initialization")
    parser.add_argument("--model_type", type=str, default="original", choices=["original", "fine_tunned"], help="Model type: 'kaf' for Siamese, 'relu' for SiameseATTReLU")
    parser.add_argument("--att_test_path", type=str, default="/home/ubuntu/project/att_faces", help="Path to AT&T test dataset folder")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use")

    args = parser.parse_args()

    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform
    data_transform = transforms.ToTensor()

    # Load the appropriate test dataset
    if args.dataset == "omniglot":
        test_set = OmniglotTest(args.test_path, transform=data_transform, times=args.times, way=args.way)
    else:
        test_set = ATTTest(args.att_test_path, transform=data_transform, times=args.times, way=args.way)

    # Load the model
    if args.model_type == "original":
        net = Siamese()
    else:
        net = SiameseATTReLU()

    state_dict = torch.load(args.model_path, map_location=device)

    # Check if keys are prefixed with "module."
    first_key = list(state_dict.keys())[0]
    if first_key.startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict

    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    datas = test_set.datas
    num_classes = test_set.num_classes

    def get_random_image_from_class(c):
        return random.choice(datas[c])

    # Create 5 same-class pairs
    same_class_pairs = []
    for _ in range(5):
        c = random.randint(0, num_classes - 1)
        img1 = get_random_image_from_class(c)
        img2 = get_random_image_from_class(c)
        same_class_pairs.append((img1, img2, 1.0))

    # Create 5 different-class pairs
    diff_class_pairs = []
    for _ in range(5):
        c1 = random.randint(0, num_classes - 1)
        c2 = random.randint(0, num_classes - 1)
        while c1 == c2:
            c2 = random.randint(0, num_classes - 1)
        img1 = get_random_image_from_class(c1)
        img2 = get_random_image_from_class(c2)
        diff_class_pairs.append((img1, img2, 0.0))

    # Combine them
    all_pairs = same_class_pairs + diff_class_pairs
    random.shuffle(all_pairs)

    print("Testing on 10 random examples (5 same-class pairs and 5 different-class pairs):")
    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(all_pairs, start=1):
            img1_t = data_transform(img1).unsqueeze(0).to(device)
            img2_t = data_transform(img2).unsqueeze(0).to(device)

            output = net(img1_t, img2_t).cpu().numpy()
            # Apply sigmoid to interpret as probability
            prob = 1 / (1 + np.exp(-output))
            pred_label = 1 if prob > 0.5 else 0

            print(f"Pair {i}: True Label={label}, Model output={output[0][0]:.4f}, Probability={prob[0][0]:.4f}, Predicted={pred_label}")

            # Convert tensors to numpy for visualization
            img1_np = img1_t.squeeze().cpu().numpy()
            img2_np = img2_t.squeeze().cpu().numpy()

            # Plot both images side by side
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            axes[0].imshow(img1_np, cmap='gray')
            axes[0].set_title("Image 1")
            axes[0].axis('off')

            axes[1].imshow(img2_np, cmap='gray')
            axes[1].set_title("Image 2")
            axes[1].axis('off')

            fig.suptitle(f"True: {label}, Pred: {pred_label}, Prob: {prob[0][0]:.4f}")
            plt.tight_layout()

            # Save the figure instead of showing it
            plt.savefig(f"results/pair_{i}.png")
            plt.close()
