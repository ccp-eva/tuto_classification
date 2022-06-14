import os
import torch
print('PyTorch version: ', torch.__version__)
from torchvision import models, transforms
import copy
import time
import datetime
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image
from matplotlib import cm
from argparse import ArgumentParser
from utils import *

'''
Definition of the CNN model
'''
class cnn_model():
    def __init__(self, model_wts_path, class_names, device='cuda:0'):
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        # Set the last layer lenght to the number of classes
        self.model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        self.model = self.model.to(device)
        model_wts = torch.load(model_wts_path)
        self.model.load_state_dict(model_wts)
        self.class_names = class_names
        self.transpose = transforms.Compose([
            transforms.Resize([224,224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.device = device

    def get_class_from_image(self, img):
        with torch.no_grad():
            # Convertion from opencv to PIL
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)

            im_pil = self.transpose(im_pil)
            im_pil = im_pil.to(self.device)
            im_pil = im_pil.unsqueeze(0)
            # img = self.transpose(img)
            # img = img.to(self.device)
            # img = img.unsqueeze(0)

            outputs = self.model(im_pil)
            outputs_normalized = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            score = outputs[0][preds[0]].item()
            score_normalized = outputs_normalized[0][preds[0]].item()
            pred = self.class_names[preds[0]]

            return pred, score_normalized

# Main
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the model to use. ')    
    parser.add_argument(
        '--video-input',
        type=str,
        default='Camera',
        help='Video path or camera to process. ')
    parser.add_argument(
        '--root-output',
        type=str,
        default='run_app_output',
        help='Root for saving all outputs. ')
    parser.add_argument(
        '--save-json-file',
        default=True,
        type=bool,
        help='Default save json results. ')
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Default save frames. ')
    parser.add_argument(
        '--save-demo-images',
        action='store_true',
        help='Default save demo images. ')
    parser.add_argument(
        '--show-demo',
        action='store_true',
        help='Show frames. ')
    parser.add_argument(
        '--device',
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help='Device for inference. ')

    print('Initialisation')
    args = parser.parse_args()

    # Create outputs folder
    os.makedirs(args.root_output, exist_ok=True)

    # Create a session names for saving files
    session_path = os.path.join(args.root_output, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(session_path)

    if args.save_json_file:
        json_file = os.path.join(session_path, 'app_results.json') 
        with open(json_file, 'w') as f:
            json.dump([], f)

    if args.save_frames:
        frames_path = os.path.join(session_path, 'frames')
        os.makedirs(frames_path)
    
    if args.save_demo_images:
        demo_images_path = os.path.join(session_path, 'demo_images')
        os.makedirs(demo_images_path)

    log_file = os.path.join(session_path, 'log.log')
    log = setup_logger('my_log', log_file)

    print_and_log("Run on %s" % (args.device), log=log)
    print_and_log('Nb of threads for OpenCV : %d' % (cv2.getNumThreads()) , log=log)
    print_and_log('%s' % (args) , log=log)

    # Classification model
    # load model weights
    class_names = np.load(os.path.join(args.model_path, 'class_names.npy')).tolist()
    classification_model = cnn_model(os.path.join(args.model_path, 'resnet18_finetuned_loss.pth'), class_names, device=args.device)
    print_and_log('CNN model loaded', log)

    # Video source - can be camera index number given by 'ls /dev/video*
    # or can be a video file, e.g. '~/Video.avi'
    if args.video_input == 'Camera':
        video_path = 0
        print_and_log('\nProcessing webcam %d' % (video_path), log=log)
    else:
        video_path = args.video_input # local test video
        print_and_log('\nProcessing video %s' % (video_path), log=log)

    cap = cv2.VideoCapture(video_path)
    start_time = time.time()
    idx = 0
    idx_notgrabbed = 0
    idx_classified = 0
    total_time_class = 0
    total_time_save_image = 0
    total_time_demo = 0
    list_results = []
    colormap = cm.get_cmap('gist_rainbow', len(class_names))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not grabbed, we wait and try again for a while
        if not ret:
            idx_notgrabbed += 1
            if idx_notgrabbed > 500:
                print_and_log('\nVideo ended or webcam disconnected', log=log)
                break
            print_and_log('Could not read frame', log=log)
            time.sleep(0.01)
            continue

        # Reset the number of not grabbed frames
        idx_notgrabbed = 0

        # Classification
        tmp_time = time.time()
        idx_classified+=1
        pred, score = classification_model.get_class_from_image(frame)
        total_time_class += time.time() - tmp_time

        json_data = {
            'time': str(datetime.datetime.now()),
            'image_id': idx,                
            'classification': {
                'pred': pred,
                'score': round(float(score),3)
            }}
        list_results.append(json_data)

        with open('current_detection.json', 'w') as f:
            json.dump(list_results, f)
                    
        if args.save_json_file:
            write_json(list_results, json_file)

        if args.save_frames:
            tmp_time = time.time()
            # Save image
            cv2.imwrite(os.path.join(frames_path, '%08d.jpg' % (idx)), frame)
            total_time_save_image += time.time() - tmp_time

        # Img with classification
        if args.save_demo_images or args.show_demo:
            tmp_time = time.time()
            img_vis = frame.copy()
            # Pick a color for each class
            colormap_idx = class_names.index(pred)
            color = [int(255*i) for i in colormap(colormap_idx)[:3]]
            # Write text on image
            cv2.putText(img_vis, '%s (%.3f)' % (pred, score), ((img_vis.shape[1]//2,img_vis.shape[0]//2)), 0, 1, color, 2)
            if args.save_demo_images:
                cv2.imwrite(os.path.join(demo_images_path, '%08d.png' % (idx)), img_vis)
            if args.show_demo:
                cv2.imshow('Demo images (press q to stop)', img_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            total_time_demo += time.time() - tmp_time
        idx+=1

    time_elapsed = time.time() - start_time
    print_and_log("%d frames processed in %.2fs -> %.2ffps" % (idx, time_elapsed, idx/time_elapsed), log=log)
    if total_time_class > 0:
        print_and_log(
            "Classification: for the video %.2fs -> %.2ffps, for the classifications (%d) %.2fs per classification -> %.2ffps " % (
                total_time_class,
                idx/total_time_class,
                idx_classified,
                total_time_class/idx_classified,
                idx_classified/total_time_class),
            log=log)

    if total_time_save_image > 0:
        print_and_log("Save image: %.2fs -> %.2ffps" % (total_time_save_image, idx/total_time_save_image), log=log)
    if total_time_demo > 0:
        print_and_log("Demo images: %.2fs -> %.2ffps" % (total_time_demo, idx/total_time_demo), log=log)
    print_and_log("\nThe program finished without error.", log=log)
    close_log(log)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
