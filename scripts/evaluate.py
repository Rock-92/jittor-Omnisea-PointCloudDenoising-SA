import sys

from _common import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import main


if __name__ == "__main__":
    main()
