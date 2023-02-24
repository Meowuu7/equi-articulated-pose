import torch
from vgtk.point3d import PointSet

class SphericalPointCloud():
	def __init__(self, xyz, feats, anchors):
		self._xyz = PointSet(xyz)
		self._feats = feats
		self._anchors = anchors

	@property
	def xyz(self):
		return self._xyz.data

	@property
	def feats(self):
		return self._feats

	@property
	def anchors(self):
		return self._anchors


class SphericalPointCloudPose():
	def __init__(self, xyz, feats, anchors, pose): # N x 3 x 3 --- rotation matrix (the pose information for each point)
		self._xyz = PointSet(xyz)
		self._feats = feats
		self._anchors = anchors
		self._pose = pose

	@property
	def xyz(self):
		return self._xyz.data

	@property
	def feats(self):
		return self._feats

	@property
	def anchors(self):
		return self._anchors

	@property
	def pose(self):
		return self._pose





