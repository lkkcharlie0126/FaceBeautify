
encode_args_default = {
    'data_dir': 'data',
    'mask_dir': 'masks',
    'load_last': '',
    'dlatent_avg': '',
    'model_url': 'gdrive:networks/stylegan2-ffhq-config-f.pkl',
    'model_res': 1024,
    'batch_size': 1,
    'optimizer': 'ggt',
    'vgg_url': 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
    'image_size': 256,
    'resnet_image_size': 256,
    'lr': 0.25,
    'decay_rate': 0.9,
    'iterations': 100,
    'decay_steps': 4,
    'early_stopping': True,
    'early_stopping_threshold': 0.5,
    'early_stopping_patience': 10,
    'load_effnet': 'data/finetuned_effnet.h5',
    'load_resnet': 'data/finetuned_resnet.h5',
    'use_preprocess_input': True,
    'use_best_loss': True,
    'average_best_loss': 0.25,
    'sharpen_input': True,
    'use_vgg_loss': 0.4,
    'use_vgg_layer': 9,
    'use_pixel_loss': 1.5,
    'use_mssim_loss': 200,
    'use_lpips_loss': 100,
    'use_l1_penalty': 0.5,
    'use_discriminator_loss': 0.5,
    'use_adaptive_loss': False,
    'randomize_noise': False,
    'tile_dlatents': False,
    'clipping_threshold': 2.0,
    'load_mask': False,
    'face_mask': True,
    'use_grabcut': True,
    'scale_mask': 1.4,
    'composite_mask': True,
    'composite_blur': 8,
    'video_dir': 'videos',
    'output_video': False,
    'video_codec': 'MJPG',
    'video_frame_rate': 24,
    'video_size': 512,
    'video_skip': 1,
}

class EncodeArgs:
    def __init__(
            self, 
            src_dir, 
            generated_images_dir,
            dlatent_dir,
            **kwargs
        ):
        self.src_dir = src_dir
        self.generated_images_dir = generated_images_dir
        self.dlatent_dir = dlatent_dir
        self.__dict__.update(kwargs)

