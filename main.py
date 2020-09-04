import os
import argparse
import numpy
import torch
import yaml
import skimage

import models

def check_yml(d):
    try:
        assert os.path.isfile(d)
        return d
    except Exception:
        raise argparse.ArgumentTypeError(
            "yml file {} cannot be located.".format(d))

def main():
    parser = argparse.ArgumentParser(description='Project Name')
    parser.add_argument('yml', nargs='?',
                        help='yml file name')

    args = parser.parse_args()

    check_yml(args.yml)
    with open(args.yml) as f:
        config = yaml.load(f, yaml.SafeLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = skimage.io.imread(config['img_path']).transpose(2,0,1)
    img = img/255.
    img += 50/255*numpy.random.randn(*img.shape)
    img-=numpy.array([0.485, 0.456, 0.406])[:,None,None]
    img/=numpy.array([0.229, 0.224, 0.225])[:,None,None]
    img = torch.tensor(img).float()[None,:,:,:]

    model = models.VGG16Features(device = device, target_layer = config['target_layer'])
    model.visualize(img, config['save_path'])


if __name__ == '__main__':
    main()
