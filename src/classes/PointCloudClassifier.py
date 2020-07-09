import json
import numpy as np
import pandas as pd
import pdal
import warnings
import tempfile
import pickle
import joblib

class PointCloudClassifier:

    def __init__(self, model=None, json_pipe=None):
        self.classifier = model
        if json_pipe is None:
            try:
                self.json_pipe = self.classifier.pdal_pipe_
            except AttributeError:
                warnings.warn("To ensure your models features are available you should supply a json_pipe or data with the appropriate features.")
        else:
            self.json_pipe = json_pipe

    def prepare_data(self, data):
        with tempfile.NamedTemporaryFile(suffix='.npy') as temp:
            np.save(temp.name, data)
            json_pipe = json.loads(self.json_pipe)
            if type(json_pipe['pipeline'][0]) is str:
                json_pipe['pipeline'][0] = temp.name
            else:
                json_pipe['pipeline'].insert(0, temp.name)
            print(json_pipe)
            pdal_pipe = pdal.Pipeline(json.dumps(json_pipe))
            if pdal_pipe.validate():
                pdal_pipe.execute()
            self._arrays = pd.DataFrame(np.array(pdal_pipe.arrays)[0])
            try:
                self.X = self._arrays.drop([x for x in self._arrays.keys() if x not in self.classifier.feature_names_], axis=1)
            except AttributeError:
                warnings.warn("Cannot validate the features in data. May get erroneous results.")
                self.X = self._arrays

    def classify_data(self):
        self.y = self.classifier.predict(self.X)
        return self.y

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        try:
            self._classifier = pickle.load(open(value, 'rb'))
        except pickle.UnpicklingError:
            try:
                self._classifier = joblib.load(value)
            except:
                raise ValueError("Expected a pickle or joblib saved model file.") #from None

    @property
    def json_pipe(self):
        return self._json_pipe

    @json_pipe.setter
    def json_pipe(self, value):
        self._json_pipe = value

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        self._pipeline = value

    def __str__(self):
        return self._classifier.__str__()


