import os
from argparse import ArgumentParser
import shutil
import datetime
from utils import *

def split_database(database, output_path, train_percentage=.6, val_percentage=.2, test_percentage=.2, video_wise=False):
    list_categories = [f for f in os.listdir(database) if os.path.isdir(os.path.join(database,f))]
    total_samples_per_set = {'train': 0, 'validation': 0, 'test': 0}

    for category in list_categories:
        print_and_log('Processing category %s' % category, log=log)
        if video_wise:
            data_per_category = os.listdir(os.path.join(database, category))
        else:
            data_per_category = getListOfFiles(os.path.join(database, category))
            # Remove the addional information from the filename
            data_per_category = [filename[len(os.path.join(database, category)):] for filename in data_per_category]
        N = len(category_videos)
        for dataset in total_samples_per_set.keys():
          os.makedirs(os.path.join(output_path, dataset, category), exist_ok=True)
        for idx in range(int(train_percentage*N)):
            shutil.copytree(os.path.join(database, category, data_per_category[idx]), os.path.join(output_path, 'train', category, data_per_category[idx]))
        for idx in range(int(val_percentage*N), int(2/3*N)):
            shutil.copytree(os.path.join(database, category, data_per_category[idx]), os.path.join(output_path, 'validation', category, data_per_category[idx]))
        for idx in range(int(test_percentage*N), N):
            shutil.copytree(os.path.join(database, category, data_per_category[idx]), os.path.join(output_path, 'test', category, data_per_category[idx]))
    print_and_log('Splitted done and saved in %s' % (output_path), log=log)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--database',
        default='my_dataset',
        type=str,
        help='Video folder with category folders. ')

    log = setup_logger('my_log', 'split_database_%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    print_and_log('Initialisation', log=log)
    args = parser.parse_args()

    # Split the database framewisely
    framewise_split = args.database + '_framewise_split'
    split_database(args.database, videowise_split)

    # Split the database videowisely
    videowise_split = args.database + '_videowise_split'
    split_database(args.database, videowise_split, train_percentage=1/3, val_percentage=1/3, test_percentage=1/3, video_wise=True)

    plot_data_distribution([framewise_split, videowise_split])


if __name__ == '__main__':
    main()
