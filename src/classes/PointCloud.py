import csv
import sys
import warnings

import laspy
import numpy as np
import shapely.geometry as geometry
from sklearn.cluster import DBSCAN

from ElevatedPointGroup import ElevatedPointGroup
from PDALObject import PDALObject
from PointCloudClassifier import PointCloudClassifier
import pc_tools
from classifications import class_name_to_id


class PointCloud:

    def __init__(self, infile=None, array=None, json_pipeline=None, use_adj_points=True, use_feet = False):
        self.infile = infile
        if infile is not None:
            self.pdal_object = PDALObject(infile)
            self.laspy_header = laspy.file.File(infile).header
        if array is None and infile is not None:
            if json_pipeline is not None:
                self.pdal_object.set_json_pipeline(json_pipeline)
            self.n_points = self.pdal_object.execute()
            self.arrays = self.pdal_object.arrays
            self.metadata = self.pdal_object.metadata
        elif array is None and infile is None:
            raise AttributeError("Please provide either an infile or an array")
        if array is not None and infile is not None:
            sys.stdout.write("Array and infile provided. Using array for points and infile for LAS header.")
            self.arrays = array
        elif array is not None and infile is None:
            self.arrays = array
        self.points = self.arrays
        self.n_points = len(self.points)
        self.classification = self.arrays['Classification']
        self.adj_points = self.points
        self.use_adj_points = use_adj_points
        self.use_feet = use_feet
        self.ep_groups = np.full(self.n_points, -1)
        self.class_groups = np.full(self.n_points, -1)
        self.candidate_vos = None

    def save_las(self, filename, points=None):
        outFile = laspy.file.File(filename, mode='w', header=self.laspy_header)
        if points is None:
            outFile.x = self.x
            outFile.y = self.y
            outFile.z = self.z
            outFile.raw_classification = self.classification
        else:
            outFile.x = points[:, 0]
            outFile.y = points[:, 1]
            outFile.z = points[:, 2]
        outFile.close()

    def run_pdal_pipeline(self, json_pipeline):
        self.pdal_object.set_json_pipeline(json_pipeline)
        self.n_points = self.pdal_object.execute()
        self.update_points(self.pdal_object.arrays)

    def update_points(self, array):
        self.arrays = array
        self.points = array
        self.adj_points = self.points

    def find_trees_basic(self, ht_min = 2, classify=5):
        self.tree_potential = ((self.classification == 1) & (self.arrays['HeightAboveGround'] >= ht_min)
                                & (self.arrays['Eigenvalue0'] > 0.5)
                                & (self.arrays['NumberOfReturns'] - self.arrays['ReturnNumber'] >= 1))
        if classify is not None:
            self.classification = (self.tree_potential, classify)
        return self.tree_potential

    def find_buildings_basic(self, ht_min=7, classify=True):
        self.roof_mask = ((self.classification == 1)
                          & (self.arrays['HeightAboveGround'] > ht_min)
                          & (self.arrays['Eigenvalue0'] <= .02)
                          & (self.arrays['NumberOfReturns'] == self.arrays['ReturnNumber']))
        if classify:
            self.classification = (self.roof_mask, 6)
        return self.roof_mask

    def unassign_classification(self, class_to_unassign=None):
        if class_to_unassign is None:
            self.classification[:] = 1
        else:
            self.classification[self.classification==class_to_unassign] = 1

    def find_elevated_points(self, ht_min = 40):
        if 'HeightAboveGround' in self.arrays.dtype.names:
            self.eps = self.arrays['HeightAboveGround'] > ht_min
        else:
            self.eps = pc_tools.ground_points_grid_filter(self.points[:,0:3], z_height=ht_min)

    def group_eps(self, min_samples=15, eps=20):
        clusters = pc_tools.dbscan_cluster(self.elevated_points[:, 0:3], eps=eps, min_samples=min_samples)
        self.ep_groups[self.elevated_points[:,3].astype(int)] = clusters.labels_
        for grp in set(clusters.labels_):
            if grp != -1:
                self.candidate_vos = (grp, ElevatedPointGroup(self.arrays[self.ep_groups == grp]))
        return self.candidate_vos

    def run_classifier(self):
        array = self.arrays
        if self.use_adj_points:
            array['X'] = self.adj_points[:, 0]
            array['Y'] = self.adj_points[:, 1]
            array['Z'] = self.adj_points[:, 2]
        self.classifier.prepare_data(array)
        self.classes = self.classifier.classify_data()

    def group_class(self, class_id=None, class_name=None):
        # TODO: Probably worthwhile to do some testing for each individual class to determine
        # optimal eps and min_samples
        if class_id is not None and class_name is not None:
            if class_id != class_name_to_id[class_name]:
                if class_id not in class_name_to_id.values():
                    class_id = class_name_to_id[class_name]
                    warnings.warn("Provided class id not recognized. Setting class_id according to class_name.")
                warnings.warn("Provided class name and class id do not match. Using provided class_id.")
        elif class_name is not None:
            class_id = class_name_to_id[class_name]
        elif class_id is None and class_name is None:
            raise Exception('Please provide either class_id or class_name')

        if class_id not in class_name_to_id.values():
            raise Exception("Unrecognized class_id.")
        dbs = DBSCAN(eps=20, min_samples=10)
        labels = dbs.fit_predict(self.points[self.classes == class_id][:, 0:3])
        start_group = np.max(self.class_groups)
        self.class_groups[self.classes == class_id] = [x+1+start_group if x !=-1 else -1 for x in labels]

    def group_geom(self, group_id, dims=2):
        if dims == 2:
            return [geometry.Point(x[0],x[1]) for x in self.points[self.class_groups==group_id]]
        elif dims == 3:
            return [geometry.Point(x[0], x[1], x[2]) for x in self.points[self.class_groups==group_id]]
        else:
            raise Exception("Dims must equal 2 or 3.")

    def group_height(self, group_id, method='max'):
        if method == 'max':
            return max(self.points[self.class_groups == group_id][:,2])
        elif method == 'avg':
            return sum(self.points[self.class_groups == group_id][:, 2])/ len(self.points[self.class_groups == group_id])
        else:
            raise Exception("Group height currently only supports 'max' and 'avg' methods.")

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        if type(value) == str:
            self._classifier = PointCloudClassifier(value)

    @property
    def candidate_vos(self):
        return self._candidate_vos

    @candidate_vos.setter
    def candidate_vos(self, value):
        if value is not None:
            try:
                key, value = value
            except TypeError:
                raise TypeError("Provide a key, value pair as a tuple.")
            self._candidate_vos[key] = value
        else:
            self._candidate_vos = dict()

    def save_candidate_vos(self, filename):
        with open(filename, 'w+', newline='') as vo_file:
            vo_writer = csv.writer(vo_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for cand_vo in self.candidate_vos.values():
                vo_writer.writerow([cand_vo.centroid[0], cand_vo.centroid[1], cand_vo.height, cand_vo.classification])

    @property
    def elevated_points(self):
        if self.eps is not None:
            return self.points[self.eps]
        else:
            return None

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
    def x(self):
        if self.use_adj_points:
            return self._adj_points[:, 0]
        else:
            return self.points[:, 0]

    @property
    def y(self):
        if self.use_adj_points:
            return self._adj_points[:, 1]
        else:
            return self.points[:, 1]

    @property
    def z(self):
        if self.use_adj_points:
            return self._adj_points[:, 2]
        else:
            return self._points[:, 2]

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value):
        try:
            array, value = value
            self._classification[array] = value
        except ValueError:
            self._classification = value

    def update_class_by_array(self, array, value):
        self._classification[array] = value

    @property
    def adj_points(self):
        if self._adj_points is not None:
            return self._adj_points
        elif self.points is not None:
            self.adj_points = self.points
            return self._adj_points
        else:
            return None

    @adj_points.setter
    def adj_points(self, value):
        if value is not None:
            x = value[:, 0].min()
            y = value[:, 1].min()
            z = value[:, 2].min()
            self._adj_points = np.array([value[:, 0] - x, value[:, 1] - y, value[:, 2] - z]).transpose()
        else:
            self._adj_points = None
