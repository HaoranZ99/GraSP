import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Running.")

    parser.add_argument(
        "--k", type=int, default=1, help="The number of layers. Default is 3.",
    )

    parser.add_argument(
        "--pool",
        nargs="?",
        default="multi",
        help="Pooling method. Default is multi.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio in train/val set. Default is 0.20.",
    )

    parser.add_argument(
        "--val-percentage",
        type=float,
        default=0.05,
        help="Validation percentage in training. Default is 0.05.",
    )

    parser.add_argument(
        "--val-every",
        type=int,
        default=20,
        help="Validation interval in training. Default is 20.",
    )

    parser.add_argument(
        "--use-pe",
        action="store_true",
        help="Use Random Walk positional encoding. Default is False.",
    )

    parser.add_argument(
        "--pe-dim",
        type=int,
        default=16,
        help="Positional encoding dimension. Only works when --use-pe is True. Default is 16.",
    )

    parser.add_argument(
        "--metric",
        nargs="?",
        default="ged",
        help="Metric: ged or mcs. Default is ged.",
    )

    parser.add_argument(
        "--sim-dist",
        nargs="?",
        default="dist",
        help="Whether to use sim score or plain value as distance value. Default is dist.",
    )

    parser.add_argument(
        "--dataset",
        nargs="?",
        default="AIDS700nef",
        help="Dataset name. Default is AIDS700nef.",
    )

    parser.add_argument(
        "--gnn-operator",
        nargs="?",
        default="gin",
        help="Type of GNN-Operator. Default is gin.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50000,
        help="Number of training epochs. Default is 50000.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimention in convolution layers. Default is 64.",
    )

    parser.add_argument(
        "--tensor-neurons",
        type=int,
        default=16,
        help="Neurons in tensor network layer. Default is 16.",
    )

    parser.add_argument(
        "--reduction",
        type=int,
        default=2,
        help="Reduce the hidden dimention. Default is 2.",
    )

    parser.add_argument(
        "--bottle-neck-neurons",
        type=int,
        default=16,
        help="Bottle neck layer neurons. Default is 16.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of graph pairs per batch. Default is 128.",
    )

    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout probability. Default is 0."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate. Default is 0.001.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5 * 10 ** -4,
        help="Adam weight decay. Default is 5*10^-4.",
    )

    parser.add_argument(
        "--save", type=str, default=None, help="Path to save the trained model."
    )

    parser.add_argument(
        "--load", type=str, default=None, help="Path to load a pretrained model."
    )


    parser.add_argument(
        "--notify",
        dest="notify",
        action="store_true",
        help="Send notification message when the code is finished (only Linux & Mac OS support).",
    )

    parser.set_defaults(notify=False)

    return parser.parse_args()
