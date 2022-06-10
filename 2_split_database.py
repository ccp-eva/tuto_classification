import os
from argparse import ArgumentParser
import shutil
import datetime
from utils import *

def split_database(database, output_path, train_percentage=.6, val_percentage=.2, test_percentage=.2, video_wise=False, log=None):
    list_categories = [f for f in os.listdir(database) if os.path.isdir(os.path.join(database,f))]
    # total_samples_per_set = {'train': 0, 'validation': 0, 'test': 0}
    # samples_prop_per_set = {'train': train_percentage, 'validation': val_percentage, 'test': test_percentage}
    # import random
    # random.seed(0)

    # for category in list_categories:
    #     print_and_log('Processing category %s' % category, log=log)
    #     if video_wise:
    #         data_per_category = os.listdir(os.path.join(database, category))
    #     else:
    #         data_per_category = getListOfFiles(os.path.join(database, category))
    #         # Shuffle the list
    #         random.shuffle(data_per_category)
    #         # Remove the addional information from the filename
    #         data_per_category = [filename[len(os.path.join(database, category))+1:] for filename in data_per_category]
    #     N = len(data_per_category)
    #     start_range = 0
    #     for dataset in total_samples_per_set.keys():
    #         os.makedirs(os.path.join(output_path, dataset, category), exist_ok=True)
    #         for idx in range(start_range, start_range+int(samples_prop_per_set[dataset]*N)):
    #             if os.path.isfile(os.path.join(database, category, data_per_category[idx])):
    #                 os.makedirs(os.path.dirname(os.path.join(output_path, dataset, category, data_per_category[idx])), exist_ok=True)
    #                 shutil.copy(os.path.join(database, category, data_per_category[idx]), os.path.join(output_path, dataset, category, data_per_category[idx]))
    #             else:
    #                 shutil.copytree(os.path.join(database, category, data_per_category[idx]), os.path.join(output_path, dataset, category, data_per_category[idx]))
    #         start_range = start_range+int(samples_prop_per_set[dataset]*N)
    # print_and_log('Splitted done and saved in %s' % (output_path), log=log)
    plot_data_distribution(output_path, list_categories)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--database',
        default='my_dataset',
        type=str,
        help='Video folder with category folders. ')

    os.makedirs('logs', exist_ok=True)
    log = setup_logger('my_log', os.path.join('logs', 'split_database_%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    print_and_log('Initialisation', log=log)
    args = parser.parse_args()

    # Split the database framewisely
    framewise_split = args.database + '_framewise_split'
    split_database(args.database, framewise_split, train_percentage=1/2, val_percentage=1/2, test_percentage=0, log=log)

    # Split the database videowisely
    videowise_split = args.database + '_videowise_split'
    split_database(args.database, videowise_split, train_percentage=1/2, val_percentage=1/2, test_percentage=0, video_wise=True, log=log)


if __name__ == '__main__':
    main()
