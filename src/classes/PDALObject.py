import json

import numpy as np
import pdal


class PDALObject:

    def __init__(self, infile, json_pipe=None, outfile=None):
        if json_pipe is not None:
            try:
                self.set_json_pipeline(json_pipe)
            except:
                self.json_pipeline = {"pipeline": [None]}
                raise Warning("json_pipe failed to load, set to empty pipeline.")
        else:
            self.json_pipeline = {"pipeline": [None]}
        self.infile = infile
        self.outfile_set = False
        self.outfile = outfile
        self.n_points = None
        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        self.executed = False
        self.arrays = None
        self.metadata = None
        self.points = None

    def add_step(self, index=None, **kwargs):
        if self.outfile_set and index is None:
            index = -2
        if index is None:
            self.json_pipeline["pipeline"].append(kwargs)
        else:
            self.json_pipeline["pipeline"].insert(index, kwargs)
        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        try:
            self.pipeline.validate()
        except Exception as err:
            if index is None:
                self.json_pipeline["pipeline"].pop()
            else:
                self.json_pipeline["pipeline"].pop(index)
            raise err

    def set_json_pipeline(self, json_pipe):
        infile = None
        outfile = None
        try:
            if "pipeline" not in json_pipe.keys():
                json_pipe = {"pipeline": json_pipe}
        except AttributeError:
            raise AttributeError("json_pipe argument is not a json formatted dictionary.")
        except Exception as err:
            raise err

        if type(json_pipe["pipeline"][0]) is str:
            infile = json_pipe["pipeline"][0]

        if type(json_pipe["pipeline"][-1]) is str:
            outfile = json_pipe["pipeline"][-1]

        if infile is None:
            json_pipe["pipeline"].insert(0, self.infile)

        if outfile is None and self.outfile_set:
            json_pipe["pipeline"].append(self.outfile)

        # Ensure valid pipeline before actually making changes:
        try:
            pdal.Pipeline(json.dumps(json_pipe)).validate()
            self.json_pipeline = json_pipe
            if outfile is not None:
                self.outfile = outfile
            if infile is not None:
                self.infile = infile
        except Exception as err:
            raise err

    def execute(self, infile=None, outfile=None):
        if infile is not None:
            self.infile = infile

        if outfile is not None:
            self.outfile = outfile
        elif self.outfile_set:
            outfile = self.outfile

        self.pipeline = pdal.Pipeline(json.dumps(self.json_pipeline))
        try:
            self.pipeline.validate()
        except Exception as err:
            print(err, " Please inspect PDALObject.json_pipeline")
        self.n_points = self.pipeline.execute()
        self.executed = True
        self.arrays = self.pipeline.arrays[0]
        self.metadata = json.loads(self.pipeline.metadata)

    def reset(self):
        self.executed = False
        self.arrays = None
        self.metadata = None
        self.points = None

    @property
    def infile(self):
        return self._infile

    @infile.setter
    def infile(self, value):
        self.reset()
        self.json_pipeline["pipeline"][0] = value
        self._infile = value

    @property
    def outfile(self):
        return self._outfile

    @outfile.setter
    def outfile(self, value):
        self.reset()
        if value is None:
            self.outfile_set = False
        elif self.outfile_set:
            self.json_pipeline["pipeline"][-1] = value
        else:
            self.json_pipeline["pipeline"].append(value)
            self.outfile_set = True
        self._outfile = value

    def remove_outfile(self):
        self.outfile = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        if value is not None:
            try:
                self._points = np.vstack([value['X'], value['Y'], value['Z']]).transpose()
            except IndexError:
                self._points = np.vstack([value[0], value[1], value[2]]).transpose()
        else:
            self._points = None

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        self._pipeline = value

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, value):
        self.points = value
        self._arrays = value
