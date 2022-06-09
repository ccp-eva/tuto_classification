import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import gc
from utils import *
from argparse import ArgumentParser

def visualize_database(inputs, labels, save_path, nb_cols=4):
    
    fig = plt.figure()
    fig.suptitle('Database samples', fontsize=8)

    for j in range(inputs.size()[0]):
        ax = plt.subplot(inputs.size()[0]//nb_cols, nb_cols, j+1)
        ax.axis('off')                
        ax.set_title(labels[j], fontsize=6)
        imshow(inputs[j])
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    return

def imshow(inp, title=None, save_path=None, show=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.pause(0.1) # pause a bit so that plots are updated
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close('all')


if __name__ == '__main__':
    
    # Initialisation
    parser = ArgumentParser()
    parser.add_argument(
        'database',
        type=str,
        help='Video folder with category folders (no splitted), splitted folder is inferred by adding _split. ')
    args = parser.parse_args()

    session_path = os.path.join('plot_database_output', args.database)
    os.makedirs(session_path, exist_ok=True)

    numb_examples = 20
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.database, x), transforms.ToTensor()) for x in ['train', 'validation', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=numb_examples, shuffle=True) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    
    print('Classes: %s' % (', '.join(class_names)))

    inputs, classes = next(iter(dataloaders['train']))
    visualize_database(inputs.cpu().data, [class_names[x] for x in classes], os.path.join(session_path, 'samples_train.png'))
   
    inputs, classes = next(iter(dataloaders['validation']))
    visualize_database(inputs.cpu().data, [class_names[x] for x in classes], os.path.join(session_path, 'samples_validation.png'))

    inputs, classes = next(iter(dataloaders['test']))
    visualize_database(inputs.cpu().data, [class_names[x] for x in classes], os.path.join(session_path, 'samples_test.png'))
    