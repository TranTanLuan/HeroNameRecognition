import sys
from utils import create_pkl_data

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir_pkl = sys.argv[2]
    create_pkl_data(in_dir, out_dir_pkl)