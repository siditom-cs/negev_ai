import argparse


parser = argparse.ArgumentParser(description='Negev AI Challenge 2022 - Team9 - HyperParameters')
parser.add_argument('--train_path', type=str, default='./data/train',
                    help='location of the data corpus')
parser.add_argument('--validation_path', type=str, default='./data/val',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--tile_dim', type=int, default=256,
                    help='Image Hight - power of 2')
parser.add_argument('--image_hight', type=int, default=256,
                    help='Image Hight - power of 2')
parser.add_argument('--image_width', type=int, default=256,
                    help='Image Width - power of 2')
parser.add_argument('--save_path', type=str, default='Model_Test',
                    help='Where to save the trained model parameters and logs')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model')

args = parser.parse_args()

