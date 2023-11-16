import argparse
import model_utils


def main():
    parser = argparse.ArgumentParser(
        description='Use this script to train a model used for classifying flowers',
        add_help=True
    )

    parser.add_argument(
        '--save_dir',
        default='./model_checkpoints',
        action='store',
        type=str,
        help="Directory path for saving checkpoints"
    )

    parser.add_argument(
        '--arch',
        action='store',
        default='densenet121',
        type=str,
        help="The architecture for the pretrained network on top of which to train classifier"
    )

    parser.add_argument(
        '--learning_rate',
        action='store',
        default=0.003,
        type=float,
        help="The learning rate for backpropagation"
    )

    parser.add_argument(
        '--hidden_units',
        default=256,
        action='store',
        type=int,
        help="The hidden units of the classifier"
    )

    parser.add_argument(
        '--epochs',
        default=2,
        action='store',
        type=int,
        help="How many epochs should the classifier iterate"
    )

    parser.add_argument(
        '--gpu',
        default=True,
        action='store_true',
        help="If passed, will use the GPU"
    )

    parser.add_argument(
        'data_dir',
        action='store',
        type=str,
        help="Data directory path"
    )

    results = parser.parse_args()

    model_utils.train_model(
        arch=results.arch,
        data_dir=results.data_dir,
        save_dir=results.save_dir,
        learning_rate=results.learning_rate,
        hidden_units=results.hidden_units,
        epochs=results.epochs,
        gpu=results.gpu
    )


if __name__ == '__main__':
    main()
