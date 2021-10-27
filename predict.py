# All file imports
import argparse


def parse_args():
    '''parse the arguments for the predict application'''
    parser = argparse.ArgumentParser(
        description='Classify an image using your neural network')
    parser.add_argument('image_path', type=str, required=True,
                        help='path to the input image to classifier (required)')
    parser.add_argument('checkpoint', type=str, required=True,
                        help='path to the model checkpoint (required)')
    parser.add_argument('--top_k', type=int,
                        help='number of top classes to show (default 5)')
    parser.add_argument('--category_names', type=str,
                        help='json file for category names')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use GPU for prediction when available')

    # parse arguments
    args = parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
