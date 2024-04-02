import os
import sys
sys.path.append('/app')

import base64
import uuid


from beautify_pipeline.aligned_image.aligned_image import align

from beautify_pipeline.encode_image.encode_image import encode
from beautify_pipeline.encode_image.encode_args import encode_args_default, EncodeArgs

from beautify_pipeline.fit_face.fit_face import fit
from beautify_pipeline.fit_face.fit_face_args import FitFaceArgs


current_dir = os.path.dirname(os.path.realpath(__file__))   
root_dir = os.path.dirname(os.path.dirname(current_dir))

def get_dirs(folder):
    dirs = {
        'raw_images_dir': os.path.join(folder, 'raw_images'),
        'aligned_images_dir': os.path.join(folder, 'aligned_images'),
        'alignement_vector_dir': os.path.join(folder, 'alignement_vector'),
        'latent_representations_dir': os.path.join(folder, 'latent_representations'),
        'masks_dir': os.path.join(folder, 'masks'),
        'generated_images_dir': os.path.join(folder, 'generated_images'),
        'output_dir': os.path.join(folder, 'out')
    }

    for dir in dirs.values():
        os.makedirs(dir, exist_ok=True)
    return dirs

def remove_dirs(dirs):
    for dir in dirs.values():
        os.rmdir(dir)


def save_base64_image(image, save_path) -> str:
    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(image))

def load_image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def pipeline(image: str) -> str:

    image_name = 'beautified_image'
    folder_name = str(uuid.uuid4())
    print(folder_name)
    dirs = get_dirs("assets/" + folder_name)

    # save image to disk
    print('saving image...')
    save_base64_image(image, os.path.join(dirs['raw_images_dir'], image_name + '.png'))

    # align image
    print('aligning image...')
    align(dirs['raw_images_dir'], dirs['aligned_images_dir'], dirs['alignement_vector_dir'])

    # encode image
    print('encoding image...')
    encode_args = encode_args_default.copy()
    encode_args['vgg_url'] = 'https://rolux.org/media/stylegan/vgg16_zhang_perceptual.pkl'
    encode_args['lr'] = 0.4
    encode_args['iterations'] = 200
    encode_args['use_best_loss'] = False
    encode_args['early_stopping'] = False
    encode_args['load_resnet'] = False
    encode_args['mask_dir'] = dirs['masks_dir']
    encode_args = EncodeArgs(
        dirs['aligned_images_dir'], 
        dirs['generated_images_dir'],
        dirs['latent_representations_dir'],
        **encode_args
    )
    encode(encode_args)

    # fit face
    print('fitting face...')
    fit_face_args = FitFaceArgs(
        src_file=os.path.join(dirs['raw_images_dir'], image_name + '.png'),
        dst_file=os.path.join(dirs['output_dir'], image_name + '.png'),
        face_path=dirs['generated_images_dir'],
        mask_path=dirs['masks_dir'],
        face_landmarks_path=dirs['alignement_vector_dir']
    )
    fit(fit_face_args)

    # load image to base64
    print(('transforming image to base64...'))
    beautified_image = load_image_to_base64(os.path.join(dirs['output_dir'], image_name + '.png'))

    print('Done!')
    return beautified_image