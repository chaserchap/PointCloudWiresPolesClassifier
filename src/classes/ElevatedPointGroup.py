import numpy as np
import scipy

from pc_tools import geometric_median

class ElevatedPointGroup:

    def __init__(self, arrays, remove_outliers=False, outlier_scaling_factor=1.0):
        self.arrays = arrays
        self.points = arrays
        self.classification = "Unknown"
        self._centroid = None
        self.outliers_removed = False
        if remove_outliers:
            self.remove_outliers(scaling_factor=outlier_scaling_factor)

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self._arrays = value

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        if value is not None:
            point_index = np.arange(0, len(value))
            try:
                self._points = np.vstack([value['X'], value['Y'], value['Z'], point_index]).transpose()
            except IndexError:
                self._points = np.vstack([value[:, 0], value[:, 1], value[:, 2], point_index]).transpose()
        else:
            self._points = None

    @property
    def centroid(self):
        if self._centroid is None:
            # Currently using a 3d geometric mean to determine the xy centroid
            # thought is that most vos will be taller, rather than wide, and that by
            # including the z component it will ensure a more centered point.
            self._centroid = geometric_median(self.points)[0:2]
        return self._centroid

    @centroid.setter
    def centroid(self):
        raise Warning("Centroid cannot be directly set.")

    @property
    def height(self):
        return max(self.points[:, 2])

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value):
        self._classification = value

    def remove_outliers(self, scaling_factor=1.0):
        #Default scaling factor to 1.0 to pick up the densest points. Need to test.
        if not self.outliers_removed:
            geo_med = geometric_median(self.points)
            dists = [x[0] for x in scipy.spatial.distance.cdist(self.points, [geo_med])]
            median = np.median(dists)
            inliers = [x[0] for x in np.argwhere(dists <= scaling_factor * median)]
            self.points = self.points[inliers]
            self.outliers_removed = True
