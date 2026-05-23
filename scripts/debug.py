from _common import with_default_task
from src.runner import main


if __name__ == "__main__":
    main(with_default_task("configs/task/debug.yaml"))
