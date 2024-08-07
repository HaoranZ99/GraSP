from trainer import Trainer
from param_parser import parameter_parser


def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a GraSP model.
    """
    args = parameter_parser()
    trainer = Trainer(args)

    if args.load:
        trainer.load()
    else:
        trainer.fit()
    trainer.score()
    if args.save:
        trainer.save()

    if args.notify:
        import os
        import sys

        if sys.platform == "linux":
            os.system('notify-send SimGNN "Program is finished."')
        elif sys.platform == "posix":
            os.system(
                """
                    osascript -e 'display notification "GraSP" with title "Program is finished."'
                """
            )
        else:
            raise NotImplementedError("No notification support for this OS.")


if __name__ == "__main__":
    main()
