import argparse
import model_utils


def main():
    parser = argparse.ArgumentParser(
        description='Use this script to predict an image class using a saved model',
        add_help=True
    )

    parser.add_argument(
        '--top_k',
        default=3,
        action='store',
        type=int,
        help="Top K most likely classes"
    )

    parser.add_argument(
        '--category_names',
        default='cat_to_name.json',
        action='store',
        type=str,
        help="Path to a JSON file containing the mapping between numeric category and class names"
    )

    parser.add_argument(
        '--gpu',
        default=True,
        action='store_true',
        help="If passed, will use the GPU"
    )

    parser.add_argument(
        'path_to_image',
        action='store',
        type=str,
        help="Path to an image for which to predict class"
    )

    parser.add_argument(
        'checkpoint',
        action='store',
        type=str,
        help="Path to a checkpoint file containing a trained model"
    )

    results = parser.parse_args()

    model_utils.predict(
        model=model_utils.load_model(results.checkpoint),
        path_to_image=results.path_to_image,
        gpu=results.gpu,
        top_k=results.top_k
    )


if __name__ == '__main__':
    main()