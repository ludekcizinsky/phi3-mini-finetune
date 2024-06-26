import os

import rootutils

# Root directory of the project
ROOT_DIR = rootutils.find_root(search_from=__file__, indicator=".project-root")
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "output")
