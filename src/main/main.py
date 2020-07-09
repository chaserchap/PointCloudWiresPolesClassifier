import pandas as pd
import numpy as np
from itertools import combinations
import joblib
from src.tools.pc_tools import geometric_median, point_to_line_dist, dbscan_cluster, build_box, whats_in_the_box, plot_box, check_quadrants
from sklearn.preprocessing import StandardScaler

def main(pc_file):
    data