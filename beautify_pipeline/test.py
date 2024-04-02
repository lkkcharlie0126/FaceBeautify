import base64
import sys
sys.path.append('/app')

from beautify_pipeline.pipeline import pipeline

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def main():
    image_path = '/app/raw_images_old/test.jpg'
    encoded_string = image_to_base64(image_path)
    pipeline(encoded_string)

    image_path = '/app/raw_images_old/11.jpg'
    encoded_string = image_to_base64(image_path)
    pipeline(encoded_string)

if __name__ == '__main__':
    main()