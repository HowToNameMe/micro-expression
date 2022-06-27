import matplotlib
matplotlib.use('Agg')
import sys
import yaml
import os
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import pandas as pd
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
from modules.bg_motion_predictor import BGMotionPredictor
from modules.fg_motion_predictor import FGMotionPredictor

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             num_kps=config['model_params']['common_params']['num_kps'],
                             **config['model_params']['avd_network_params'])
    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        print("create BGMotionPredictor")
        bg_predictor = BGMotionPredictor()
        bg_predictor.to(device)
        bg_predictor.eval()

    fg_predictor = None
    if (config['model_params']['common_params']['fg']):
        print("create FGMotionPredictor")
        fg_predictor = FGMotionPredictor()
        fg_predictor.to(device)
        fg_predictor.eval()
    
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)
       
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])
    if 'bg_predictor' in checkpoint:
        bg_predictor.load_state_dict(checkpoint['bg_predictor'])
    if 'fg_predictor' in checkpoint:
        fg_predictor.load_state_dict(checkpoint['fg_predictor'])
    
    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()
    
    return inpainting, kp_detector, dense_motion_network, avd_network, bg_predictor, fg_predictor


def make_animation(source_image, source_image_mask, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, bg_predictor, fg_predictor, device, mode = 'relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        source_mask = torch.tensor(source_image_mask[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source_mask = source_mask.to(device)
        
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            
            bg_param = None
            if bg_predictor!=None:
                bg_param = bg_predictor(source, driving_frame)
                # print(bg_param)
            
            fg_param = None
            if fg_predictor!=None:
                fg_param = fg_predictor(source, driving_frame)
                # print(fg_param)

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = bg_param, fg_param = fg_param,
                                                    dropout_flag = False)
            out = inpainting_network(source, source_mask, dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= 'cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--src_drv_csv", default='./dataset/Mixed_dataset/Mixed_dataset_test.csv', help="path to csv file")
    parser.add_argument("--result_video", default=None, help="path to output")
    
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'], help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")
    
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    opt = parser.parse_args()

    data = pd.read_csv(opt.src_drv_csv)
    os.makedirs(opt.result_video, exist_ok=True)

    for i in range(len(data['driving'])):
        src_dir = data['source'][i]

        src_img = os.listdir(os.path.join("./dataset/Mixed_dataset/test",src_dir))[0]
        source_image = imageio.imread(os.path.join("./dataset/Mixed_dataset/test",src_dir,src_img))

        if os.path.exists(os.path.join("./dataset/Mixed_dataset/test_mask",src_dir)):
            src_img_mask = os.listdir(os.path.join("./dataset/Mixed_dataset/test_mask",src_dir))[0]
            source_image_mask = imageio.imread(os.path.join("./dataset/Mixed_dataset/test_mask",src_dir,src_img_mask))
            source_image_mask = source_image_mask[:,:,None]
        else:
            source_image_mask = np.zeros((256,256,1))

        drv_vid = data['driving'][i]
        driving_video = []
        drv_vid_files = os.listdir(os.path.join("./dataset/Mixed_dataset/test",drv_vid))

        f = lambda x:x.split('.')[-1] in ['png','jpg','bmp']
        drv_vid_files = list(filter(f, drv_vid_files))
        drv_vid_files.sort()

        for im in drv_vid_files:
            driving_video.append(imageio.imread(os.path.join("./dataset/Mixed_dataset/test",drv_vid,im)))
    
        if opt.cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        
        source_image = resize(source_image, opt.img_shape)[..., :3]
        source_image_mask = resize(source_image_mask, opt.img_shape)[..., :1]
        driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
        inpainting, kp_detector, dense_motion_network, avd_network, bg_predictor, fg_predictor = \
            load_checkpoints(config_path = opt.config, checkpoint_path = opt.checkpoint, device = device)
    
        if opt.find_best_frame:
            i = find_best_frame(source_image, driving_video, opt.cpu)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = opt.mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, source_image_mask, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, bg_predictor, fg_predictor, device = device, mode = opt.mode)
        
        print(drv_vid+'_'+src_dir+'.mp4')
        imageio.mimsave(os.path.join(opt.result_video, drv_vid+'_'+src_dir+'.mp4'), [img_as_ubyte(frame) for frame in predictions], fps=100)

