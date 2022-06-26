from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import GeneratorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from frames_dataset import DatasetRepeater
import math

def train(config, inpainting_network, kp_detector, bg_predictor, fg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    train_params = config['train_params']
    optimizer = torch.optim.Adam(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()) +
                    list(kp_detector.parameters()), 'initial_lr': train_params['lr_generator']}],lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)
    
    optimizer_bg_predictor = None
    param_bg_fg = []
    if bg_predictor: 
        print("bg_predictor created")
        param_bg_fg += list(bg_predictor.parameters())
    
    if fg_predictor:
        print("fg_predictor created")
        param_bg_fg += list(fg_predictor.parameters())

    optimizer_bg_predictor = torch.optim.Adam(
        [{'params':param_bg_fg,'initial_lr': train_params['lr_generator']}],
        lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network = inpainting_network, dense_motion_network = dense_motion_network,       
            kp_detector = kp_detector, bg_predictor = bg_predictor, fg_predictor = fg_predictor,
            optimizer = optimizer, optimizer_bg_predictor = optimizer_bg_predictor)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    scheduler_optimizer = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    if bg_predictor or fg_predictor:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'],
                                              gamma=0.1, last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    print("length of the dataset is {}".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, 
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, bg_predictor, fg_predictor, dense_motion_network, inpainting_network, train_params)

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full).cuda()  
        
    bg_start = train_params['bg_start']
    fg_start = train_params['fg_start']
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for _, x in tqdm(enumerate(dataloader)):
                if(torch.cuda.is_available()):
                    x['driving'] = x['driving'].cuda()
                    x['source'] = x['source'].cuda()
                    x['source_mask'] = x['source_mask'].cuda()

                losses_generator, generated = generator_full(x, epoch)
                
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward()

                clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type = math.inf)
                clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type = math.inf)
                if bg_predictor and epoch>=bg_start:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type = math.inf)
                if fg_predictor and epoch>=fg_start:
                    clip_grad_norm_(fg_predictor.parameters(), max_norm=10, norm_type = math.inf)
                
                optimizer.step()
                optimizer.zero_grad()
                if (bg_predictor and epoch>=bg_start) or (fg_predictor and epoch>=fg_start):
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()
                
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_optimizer.step()
            if bg_predictor or fg_predictor:
                scheduler_bg_predictor.step()
            
            model_save = {
                'inpainting_network': inpainting_network,
                'dense_motion_network': dense_motion_network,
                'kp_detector': kp_detector,
                'optimizer': optimizer,
            }
            if bg_predictor and epoch>=bg_start:
                model_save['bg_predictor'] = bg_predictor
                model_save['optimizer_bg_predictor'] = optimizer_bg_predictor
            
            if fg_predictor  and epoch>=bg_start:
                model_save['fg_predictor'] = fg_predictor
            # print(model_save.keys(), x.size(), generated.size())
            # print(generated.keys())
            logger.log_epoch(epoch, model_save, inp=x, out=generated)

