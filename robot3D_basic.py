#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from vedo import dataurl, Mesh, Sphere, show, settings, Axes, Arrow, Cylinder, screenshot, Plotter
import numpy as np
from icecream import ic
from typing import List, Tuple


def RotationMatrix(theta, axis_name):
		""" calculate single rotation of $theta$ matrix around x,y or z
				code from: https://programming-surgeon.com/en/euler-angle-python-en/
		input
				theta = rotation angle(degrees)
				axis_name = 'x', 'y' or 'z'
		output
				3x3 rotation matrix
		"""

		c = np.cos(theta * np.pi / 180)
		s = np.sin(theta * np.pi / 180)
	
		if axis_name =='x':
				rotation_matrix = np.array([[1, 0,  0],
																		[0, c, -s],
																		[0, s,  c]])
		if axis_name =='y':
				rotation_matrix = np.array([[ c,  0, s],
																		[ 0,  1, 0],
																		[-s,  0, c]])
		elif axis_name =='z':
				rotation_matrix = np.array([[c, -s, 0],
																		[s,  c, 0],
																		[0,  0, 1]])
		return rotation_matrix


def createCoordinateFrameMesh():
		"""Returns the mesh representing a coordinate frame
		Args:
			No input args
		Returns:
			F: vedo.mesh object (arrows for axis)
			
		"""         
		_shaft_radius = 0.05
		_head_radius = 0.10
		_alpha = 1
		
		
		# x-axis as an arrow  
		x_axisArrow = Arrow(start_pt=(0, 0, 0),
												end_pt=(1, 0, 0),
												s=None,
												shaft_radius=_shaft_radius,
												head_radius=_head_radius,
												head_length=None,
												res=12,
												c='red',
												alpha=_alpha)

		# y-axis as an arrow  
		y_axisArrow = Arrow(start_pt=(0, 0, 0),
												end_pt=(0, 1, 0),
												s=None,
												shaft_radius=_shaft_radius,
												head_radius=_head_radius,
												head_length=None,
												res=12,
												c='green',
												alpha=_alpha)

		# z-axis as an arrow  
		z_axisArrow = Arrow(start_pt=(0, 0, 0),
												end_pt=(0, 0, 1),
												s=None,
												shaft_radius=_shaft_radius,
												head_radius=_head_radius,
												head_length=None,
												res=12,
												c='blue',
												alpha=_alpha)
		
		originDot = Sphere(pos=[0,0,0], 
											 c="black", 
											 r=0.10)


		# Combine the axes together to form a frame as a single mesh object 
		F = x_axisArrow + y_axisArrow + z_axisArrow + originDot
				
		return F

def homogeneous2cartesian(X_h: np.ndarray) -> np.ndarray:
		"""Converts the coordinates of a set of 3-D points from
		homogeneous coordinates to Cartesian coordinates.

		Args:
			X_h: MxN np.ndarray (float) containing N points in homogeneous coords.
					 Each point is a column of the matrix.

		Returns:
			X_c: (M-1)xN np.ndarray (float) in Cartesian coords.
					 Each point is a column of the matrix.

		"""

		# Number of rows (dimension of points).
		nrows = X_h.shape[0]

		# Divide each coordinate by the last to convert point set from homogeneous to Cartesian
		# (using vectorized calculation for speed and concise code)
		X_c = X_h[0:nrows-1,:] / X_h[-1,:]

		return X_c

def homogeneous2cartesian(X_h: np.ndarray) -> np.ndarray:
		"""Converts the coordinates of a set of 3-D points from
		homogeneous coordinates to Cartesian coordinates.

		Args:
			X_h: MxN np.ndarray (float) containing N points in homogeneous coords.
					 Each point is a column of the matrix.

		Returns:
			X_c: (M-1)xN np.ndarray (float) in Cartesian coords.
					 Each point is a column of the matrix.

		"""

		# Number of rows (dimension of points).
		nrows = X_h.shape[0]

		# Divide each coordinate by the last to convert point set from homogeneous to Cartesian
		# (using vectorized calculation for speed and concise code)
		X_c = X_h[0:nrows-1,:] / X_h[-1,:]

		return X_c

def get_rotation_and_translation_matrix(angle: float, axis_name = None) -> np.ndarray:
		""" calculate single rotation of $theta$ matrix around x,y or z
		input
				angle = rotation angle in degrees
				axis_name = 'x', 'y' or 'z'
		output
				4x4 rotation matrix
		"""

		# Convert angle from degrees to radians.
		theta = angle * np.pi / 180

		# Pre-calculate the cosine and sine values
		c = np.cos(theta)
		s = np.sin(theta)

		# Select the correct rotation matrix for each axis: x, y, or z.
		if axis_name == 'x':
				rotation_matrix = np.array([[1, 0, 0, 0],[0, c, -s, 0],[0, s, c, 0],[0, 0, 0, 1]])
		if axis_name == 'y':
				rotation_matrix = np.array([[c, 0, s, 0],[0, 1, 0, 0],[-s, 0, c, 0],[0, 0, 0, 1]])
		elif axis_name == 'z':
				rotation_matrix = np.array([[c, -s, 0, 0],[s, c, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
		elif axis_name == None:
				rotation_matrix = np.eye(4)    # Identity matrix

		return rotation_matrix

def get_rotation_and_translation_matrix(angle: float, translation : np.array, axis_name = None) -> np.ndarray:
		""" calculate single rotation of $theta$ matrix around x,y or z
		input
				angle = rotation angle in degrees
				axis_name = 'x', 'y' or 'z'
		output
				4x4 rotation matrix
		"""
		assert translation.shape == (3,) or translation.shape==(3,1),"Warning, translation not three dimensional"

		# Convert angle from degrees to radians.
		theta = angle * np.pi / 180

		# Pre-calculate the cosine and sine values
		c = np.cos(theta)
		s = np.sin(theta)

		tx : float = translation[0]
		ty : float = translation[1]
		tz : float = translation[2]

		# Select the correct rotation matrix for each axis: x, y, or z.
		if axis_name == 'x':
				rotation_matrix = np.array([[1, 0, 0, tx],[0, c, -s, ty],[0, s, c, tz],[0, 0, 0, 1]])
		if axis_name == 'y':
				rotation_matrix = np.array([[c, 0, s, tx],[0, 1, 0, ty],[-s, 0, c, tz],[0, 0, 0, 1]])
		elif axis_name == 'z':
				rotation_matrix = np.array([[c, -s, 0, tx],[s, c, 0, ty],[0, 0, 1, tz],[0, 0, 0, 1]])
		elif axis_name == None:
				rotation_matrix = np.eye(4)    # Identity matrix

		return rotation_matrix

def get_scaling_matrix(sx: float = 1, sy: float = 1, sz: float = 1) -> np.ndarray:
		""" construct a scaling matrix.
		input
				sx: scaling factor for x dimension(float)
				sy: scaling factor for y dimension(float)
				sz: scaling factor for z dimension(float)
		output
				4x4 scaling matrix  (3-D scaling in homogeneous coordinates)
		"""

		# 3-D scaling in homogeneous coordinates
		S = np.array([[sx, 0, 0, 0],[0, sy, 0, 0],[0, 0, sz, 0],[0, 0, 0, 1]])

		return S


def get_translation_matrix(tx: float = 0, ty: float = 0, tz: float = 0) -> np.ndarray:
		""" construct a scaling matrix.
		input
				tx: x-translation (float)
				ty: y-translation (float)
				tz: z-translation (float)
		output
				4x4 rotation matrix  (3-D translation in homogeneous coordinates)
		"""

		# 3-D translation in homogeneous coordinates
		T = np.array([[1, 0, 0, tx],[0, 1, 0, ty],[0, 0, 1, tz],[0, 0, 0, 1]])

		return T

def cartesian2homogeneous(X_c: np.ndarray) -> np.ndarray:
		"""Converts the coordinates of a set of 3-D points from
		Cartesian coordinates to homogeneous coordinates.

		Args:
			X_c: M x N np.ndarray (float). It contains N points in M-dimensional space.
					 Each point is a column of the matrix.

		Returns:
			X_h: (M+1) x N np.ndarray (float) in homogeneous coords. It contains N points in (M+1)-dimensional space.
					 Each point is a column of the matrix.

		"""

		# Number of columns (number of points in the set).
		ncols = X_c.shape[1]

		# Add an extra row of 1s in the matrix.
		X_h = np.block([[X_c],
									 [ np.ones((1, ncols))]])

		return X_h

def apply_transformation(H,X):
		"""transforms object using a compound transformation

		Args:
			X: 3 x N np.ndarray (float). It contains N points in 3-dimensional space
									in Cartesian coordinates. Each point is a column of the matrix.

			H: 4x4 Transformation matrix in homogeneous coordinates to be applied to the point set.

		Returns:
			Y:  3 x N np.ndarray (float). It contains N points in 3-dimensional space
											in Cartesian coordinates. Each point is a column of the matrix.

		"""
		ic(H)
		ic(X)

		assert len(H.shape) == 2, "H must be two dimensional"
		assert len(X.shape) == 2, "X must be two dimensional"
		assert H.shape[1]  - 1 == X.shape[0], f"Rows in transformation must match columns in input, {H.shape=} {X.shape=}"


		# Convert points to Homogeneous coords before transforming them
		XHom = cartesian2homogeneous(X)
		# Apply kransformation
		Y = H @ XHom

		# Convert points back to Cartesian coords before plotting
		Y = homogeneous2cartesian(Y)

		return Y

def create_frame(current_transform : np.array, color, height : float, offset : float) -> Mesh:
	# Now, let's create a cylinder and add it to the local coordinate frame
	link_mesh = Cylinder(r=0.4, 
												height=height, 
												pos = (height/2 + offset,0,0),
												c=color,
												alpha=.8, 
												axis=(1,0,0)
												)
	
	# Also create a sphere to show as an example of a joint
	r1 = 0.4
	sphere = Sphere(r=r1).pos(-r1,0,0).color("gray").alpha(.8)


	# Combine all parts into a single object 
	FrameArrows = createCoordinateFrameMesh()
	Frame = FrameArrows + link_mesh + sphere
	Frame.apply_transform(current_transform)
	return Frame

def gen_mesh_from_transformation(Li : float, current_transform : np.array, current_Li_vec : np.array, color : str, mesh : Mesh = None) -> Mesh:

	if(mesh == None):
		mesh = Cylinder

		# Now, let's create a cylinder and add it to the local coordinate frame
		link1_mesh = Cylinder(r=0.4, 
													height=Li, 
													pos = current_Li_vec/2,
													c=color, 
													alpha=.8, 
													axis=(1,0,0)
													)
	elif(mesh == Sphere):
		# Now, let's create a cylinder and add it to the local coordinate frame
		link1_mesh = Sphere(r=0.4, 
													pos = current_Li_vec,
													alpha=.8, 
													)
	
	# Combine all parts into a single object 
	Frame =  link1_mesh

	# Transform the part to position it at its correct location and orientation 
	Frame.apply_transform(current_transform)  
	return Frame
def get_end_effector(cumulative_mats : np.array, to_print : bool = False) -> Tuple[np.array, np.array]:
	cumulative_transform = np.eye(4)
	if(to_print):
		print(f"Getting final end effector")
		print(f"==============================")

		ic(cumulative_mats)
		ic(len(cumulative_mats))

	transforms = cumulative_mats.copy()
	transforms.reverse()
	for mat in transforms:
		cumulative_transform = mat @ cumulative_transform
		if(to_print):
			print(f"Multiplying by ")
			ic(mat)
			ic(cumulative_transform)
	
	if(to_print):
		ic(cumulative_transform)
	return cumulative_transform, cumulative_transform[0:3, -1]

def aggregate_frames(cumulative_mats : List[np.array]) -> List[Mesh]:

	_, end_effector = get_end_effector(cumulative_mats, to_print=False)

	current_transform : np.array = np.eye(4)
	for mat in cumulative_mats:
		current_transform = mat @ current_transform
		current_p : np.array = mat[0:3, -1]
		ic(current_p)
		print("Local position")
		local_pos = end_effector - current_p
		ic(local_pos)
		ic("end position is ")
		ic(apply_transformation(current_transform, np.expand_dims(local_pos, axis=1)))

	# ic(end_effector)
	# frames.append(Sphere(pos=neutral_Li_vec, r=r1).apply_transform(cumulative_transform))
	# frames.append(Cylinder(r=r1, pos=end_effector, height=(Li/2),axis=(1,0,0), alpha=.8, c=colors[i]).apply_transform(cumulative_transform))


def forward_kinematics(Phi : np.array, L1 : float, L2 : float, L3 : float, L4 : float):
	settings.default_backend = 'vtk'
	# begin by drawing the robot in its base form

	vp = Plotter()
	axes = Axes(xrange=(0,20), yrange=(-2,10), zrange=(0,6))
	colors : List[str] = ["yellow", "green", "red", "blue"]

	lengths : List[float] = [L1,L2,L3, L4]
	ic(lengths)
	ic(Phi)
	answers = []

	assert Phi.shape == (4, 1) or Phi.shape==(4,), f"Phi array is of improper shape! {Phi.shape}, expected {(4,1)} or {(4,)}"

	initial_matrix : np.array = np.eye(4)

	r1 = 0.4

	world_origin = np.array([3,2,0])
	joint_offset = np.array([r1, 0, 0])
	initial_matrix[0:3, -1] = world_origin
	ic(initial_matrix)

	frames : List[Mesh] = []

	cumulative_mats : List[np.array] = []
	cumulative_mats.append(initial_matrix)
	
	for i, phi in enumerate(list(Phi)):

		Li : float = lengths[i]

		print(f"Angle is {phi}")
		ic(i)
		ic(Li)
		# print(f"Generating rotation matrix for part {i}")

		# if i > 1 and i < len(Phi):
		# 	end_effector_loc += np.array([2*r1, 0, 0])

		# get current Li plus some offset due to the joint
		neutral_Li_vec = np.array([Li , 0 , 0])
		if i > 0 and i < len(Phi) - 1: 
			offset = 2 * joint_offset
		elif i == 0:
			offset =  1 * joint_offset
		else:
			offset = np.zeros(3)

		neutral_Li_vec = neutral_Li_vec + offset

		# multiply the Li vector into the last matrix
		current_transform = get_rotation_and_translation_matrix(-1 * phi, neutral_Li_vec, axis_name="z")
		ic(current_transform)
		# need to add the offset of the bottom half of the joint to the current_transform matrix

		cumulative_transform = np.eye(4)


		cumulative_transform, end_effector = get_end_effector(cumulative_mats, to_print=False)
		answers.append(cumulative_transform[0:3, -1])
		cumulative_mats.append(current_transform)
		ic(cumulative_transform)
		ic(end_effector)


	# print("Aggregating frames")
	# aggregate_frames(cumulative_mats)

	_ , e = get_end_effector(cumulative_mats, to_print=True)
	# vp.show(frames, axes, viewup="z" ,interactive=True)
	assert len(answers) == 4
	print(f"Final position is {e}")
	answers.append(e)

	# Function implementation goes here
	return tuple(answers)


def getLocalFrameMatrix(R_ij, t_ij): 
		"""Returns the matrix representing the local frame
		Args:
			R_ij: rotation of Frame j w.r.t. Frame i 
			t_ij: translation of Frame j w.r.t. Frame i 
		Returns:
			T_ij: Matrix of Frame j w.r.t. Frame i. 
			
		"""             
		# Rigid-body transformation [ R t ]
		T_ij = np.block([[R_ij,                t_ij],
										 [np.zeros((1, 3)),       1]])
		
		return T_ij
	

def main():
	settings.default_backend = 'vtk'
	# Set the limits of the graph x, y, and z ranges 
	axes = Axes(xrange=(0,20), yrange=(-2,10), zrange=(0,6))

	# vp = Plotter(offscreen=True)
	vp = Plotter()

	# Lengths of arm parts 
	L1 = 5   # Length of link 1
	L2 = 8   # Length of link 2

	# Joint angles 
	phi1 = 30     # Rotation angle of part 1 in degrees
	phi2 = -10    # Rotation angle of part 2 in degrees
	phi3 = 0      # Rotation angle of the end-effector in degrees
	
	# Matrix of Frame 1 (written w.r.t. Frame 0, which is the previous frame) 
	R_01 = RotationMatrix(phi1, axis_name = 'z')   # Rotation matrix
	p1   = np.array([[3],[2], [0.0]])              # Frame's origin (w.r.t. previous frame)
	t_01 = p1                                      # Translation vector
	
	T_01 = getLocalFrameMatrix(R_01, t_01)         # Matrix of Frame 1 w.r.t. Frame 0 (i.e., the world frame)
	
	# Create the coordinate frame mesh and transform
	Frame1Arrows = createCoordinateFrameMesh()
	
	# Now, let's create a cylinder and add it to the local coordinate frame
	link1_mesh = Cylinder(r=0.4, 
												height=L1, 
												pos = (L1/2,0,0),
												c="yellow", 
												alpha=.8, 
												axis=(1,0,0)
												)
	
	# Also create a sphere to show as an example of a joint
	r1 = 0.4
	sphere1 = Sphere(r=r1).pos(-r1,0,0).color("gray").alpha(.8)

	# Combine all parts into a single object 
	Frame1 = Frame1Arrows + link1_mesh + sphere1

	# Transform the part to position it at its correct location and orientation 
	Frame1.apply_transform(T_01)  
	
	# Matrix of Frame 2 (written w.r.t. Frame 1, which is the previous frame) 	
	R_12 = RotationMatrix(phi2, axis_name = 'z')   # Rotation matrix
	p2   = np.array([[L1],[0.0], [0.0]])           # Frame's origin (w.r.t. previous frame)
	t_12 = p2                                      # Translation vector
	
	# Matrix of Frame 2 w.r.t. Frame 1 
	T_12 = getLocalFrameMatrix(R_12, t_12)
	
	# Matrix of Frame 2 w.r.t. Frame 0 (i.e., the world frame)
	T_02 = T_01 @ T_12
	
	# Create the coordinate frame mesh and transform
	Frame2Arrows = createCoordinateFrameMesh()
	
	# Now, let's create a cylinder and add it to the local coordinate frame
	link2_mesh = Cylinder(r=0.4, 
												height=L2, 
												pos = (L2/2,0,0),
												c="red", 
												alpha=.8, 
												axis=(1,0,0)
												)
	
	# Combine all parts into a single object 
	Frame2 = Frame2Arrows + link2_mesh
	
	# Transform the part to position it at its correct location and orientation 
	Frame2.apply_transform(T_02)  
	
	# Matrix of Frame 3 (written w.r.t. Frame 2, which is the previous frame) 	
	R_23 = RotationMatrix(phi3, axis_name = 'z')   # Rotation matrix
	p3   = np.array([[L2],[0.0], [0.0]])           # Frame's origin (w.r.t. previous frame)
	t_23 = p3                                      # Translation vector
	
	# Matrix of Frame 3 w.r.t. Frame 2 
	T_23 = getLocalFrameMatrix(R_23, t_23)
	
	# Matrix of Frame 3 w.r.t. Frame 0 (i.e., the world frame)
	T_03 = T_01 @ T_12 @ T_23
	
	# Create the coordinate frame mesh and transform. This point is the end-effector. So, I am 
	# just creating the coordinate frame. 
	Frame3 = createCoordinateFrameMesh()

	# Transform the part to position it at its correct location and orientation 
	Frame3.apply_transform(T_03)  

	output_path : Path = Path("./output")

	# Show everything 
	print(f"Saving to path")
	print(output_path / Path("1").with_suffix(".png"))
	path = output_path / Path("1")
	# vp.show([Frame1, Frame2, Frame3], axes, viewup="z" ,interactive=False).screenshot(path)
	vp.show([Frame1, Frame2, Frame3], axes, viewup="z" ,interactive=True)
	screenshot("output.png")

	Frame1.apply_transform(T_01)
	path = output_path / Path("2")

	# vp.show([Frame1, Frame2, Frame3], axes, viewup="z" ,interactive=False).screenshot(path)
	vp.show([Frame1, Frame2, Frame3], axes, viewup="z" ,interactive=True)


if __name__ == '__main__':

	print("##############################")
	print("  Assertion 2")
	print("##############################")
		# main()
	# Lentghs of the parts
	L1, L2, L3, L4 = [5, 8, 3, 0]
	Phi = np.array([0, 0, 0, 0])
	T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)
	ic((T_01, T_02, T_03, T_04))
	
	actual = e
	expected = np.array([21, 2,  0. ])

	print(f"{expected=}, {actual=}")
	assert1 = np.allclose(expected, actual)
	ic(assert1)


	# print("##############################")
	# print("  Assertion 3")
	# print("##############################")

	# # Lentghs of the parts
	# L1, L2, L3, L4 = [5, 8, 3, 0]
	# Phi = np.array([-30, 50, 30, 0])
	# T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)
	
	# actual = e
	# expected = np.array([18.47772028,  4.71432837,  0. ])

	# print(f"{expected=}, {actual=}")
	# assert np.allclose(expected, actual)
