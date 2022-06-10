import os
from argparse import ArgumentParser
import datetime
from utils import *

'''
Frame Extractor from videos
'''
def frame_extractor(video_path, save_path, width=640):
    # Load Video
    cap = cv2.VideoCapture(video_path)
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    # Check if video uploaded
    if not cap.isOpened():
        sys.exit("Unable to open the video, check the path.\n")

    while frame_number < length_video:
        # Load video
        _, rgb = cap.read()

        # Check if load Properly
        if _ == 1:
            # Resizing and Save
            rgb = cv2.resize(rgb, (width, rgb.shape[0]*width//rgb.shape[1]))
            cv2.imwrite(os.path.join(save_path, '%08d.png' % frame_number), rgb)
            frame_number+=1
    cap.release()

'''
Loop over the videos in a folder to extract frames and save them in a folder
'''
def extract_frames_from_video_folder(category_path, output_path):
    list_videos = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path,f))]
    print("There are %d videos:" % len(list_videos), list_videos)
    for video in list_videos:
        print('Processing video %s' % video)
        video_path = os.path.join(category_path, video)
        save_path = os.path.join(output_path, os.path.splitext(video)[0])
        os.makedirs(save_path, exist_ok=True)
        frame_extractor(video_path, save_path)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--video_input',
        type=str,
        default='videos',
        help='Video folder with category folders. ')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='my_dataset',
        help='Output where the frames will be saved following same tree than the video folder. ')
    
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    log = setup_logger('my_log', os.path.join('logs', 'create_database_%s.log' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))

    list_categories = [f for f in os.listdir(args.video_input) if os.path.isdir(os.path.join(args.video_input,f))]
    print_and_log("There are %d categories: %s" % (len(list_categories), ', '.join(list_categories)), log=log)

    os.makedirs(args.output_folder, exist_ok=True)
    for category in list_categories:
        print_and_log('Processing category %s' % category, log=log)
        category_path = os.path.join(args.video_input, category)
        output_path = os.path.join(args.output_folder, category)
        os.makedirs(output_path, exist_ok=True)
        extract_frames_from_video_folder(category_path, output_path)

    print_and_log("Stats on your freasly created dataset", log=log)
    total_samples = 0
    for category in list_categories:
        nb_samples = 0
        for video in os.listdir(os.path.join(args.output_folder, category)):
            nb_samples += len(os.listdir(os.path.join(args.output_folder, category, video)))
        print_and_log("For %s there are %d samples" % (category, nb_samples), log=log)
        total_samples+=nb_samples
    print_and_log("Total number of samples: %d" % total_samples, log=log)

if __name__ == '__main__':
    main()
