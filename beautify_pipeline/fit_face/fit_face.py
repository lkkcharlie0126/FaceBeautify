import os
import sys
import argparse
from ffhq_dataset.face_replacement import face_replace
import bz2
from keras.utils import get_file
from ffhq_dataset.landmarks_detector import LandmarksDetector
from shutil import copyfile
import tqdm

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def fit(args):
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2', LANDMARKS_MODEL_URL, cache_subdir='temp'))
    #landmarks_model_path = unpack_bz2('models/shape_predictor_68_face_landmarks.dat.bz2')
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    #Copy file to new location to work on it
    copyfile(args.src_file, args.dst_file)

    #Replace every face that has a mask
    for f in os.listdir(args.face_path):
        if f.endswith(".png") and f.startswith(os.path.splitext(os.path.basename(args.src_file))[0]):
            filename = os.path.splitext(f)[0]
            for i, generated_face_landmarks in enumerate(landmarks_detector.get_landmarks(args.face_path + "/" + filename + ".png"), start=1):
                face_replace(args.dst_file, args.face_path + "/" + filename + ".png", args.mask_path + "/" + filename + ".png", args.face_landmarks_path + "/" + filename + ".npy", generated_face_landmarks, args.dst_file )
    print("Done!")
