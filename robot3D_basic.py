#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from argparse import ArgumentParser

from vedo import dataurl, Mesh, Sphere, show, settings, Axes, Arrow, Cylinder, screenshot, Plotter
import numpy as np
from icecream import ic
from typing import List, Tuple
from enum import Enum


class SEGMENT_TYPE(Enum):
	NONE = 3
	LOWER_JOINT = 0
	UPPER_JOINT=1
	ARM = 2

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
def get_end_effector(r1 : float, cumulative_mats : np.array, to_print : bool = False) -> Tuple[np.array, np.array]:
	cumulative_transform = np.eye(4)
	if(to_print):
		print(f"Getting final end effector")
		print(f"==============================")

		ic(cumulative_mats)
		ic(len(cumulative_mats))
	joint_offset = np.array([r1, 0, 0])

	transforms = cumulative_mats.copy()

	# transforms = transforms[::-1]
	for i, (type, mat) in enumerate(reversed(transforms)):
		cumulative_transform = mat @ cumulative_transform
		if(to_print):
			ic(i)
			print(f"Multiplying by ")
			ic(mat)
			# print(f"Cumulative is ")
			# ic(cumulative_transform)

		# if(to_print):
		# 	print(f"p{i+1} =  {end_effector}")

	
	if(to_print):
		ic(cumulative_transform)
	return cumulative_transform, cumulative_transform[0:3, -1]

def forward_kinematics(Phi : np.array, L1 : float, L2 : float, L3 : float, L4 : float, index : int = None):

	if index == None:
		index = 0

	settings.default_backend = 'vtk'
	# begin by drawing the robot in its base form

	vp = Plotter()
	axes = Axes(xrange=(-30,30), yrange=(-30,30), zrange=(0,6))
	colors : List[str] = ["yellow", "green", "red", "blue", "purple"]

	lengths : List[float] = [L1,L2,L3, L4]
	ic(lengths)
	ic(Phi)
	answers = []

	assert Phi.shape == (4, 1) or Phi.shape==(4,), f"Phi array is of improper shape! {Phi.shape}, expected {(4,1)} or {(4,)}"

	initial_matrix : np.array = np.eye(4)

	r1 = 0.4

	world_origin = np.array([3,2,0])
	joint_offset = np.array([r1, 0, 0])
	initial_matrix[0:3, -1] = world_origin - joint_offset
	ic(initial_matrix)

	frames : List[Mesh] = []

	segment_mats : List[np.array] = []
	segment_mats.append((SEGMENT_TYPE.NONE,initial_matrix))

	first_zero_index = lengths.index(0)
	ic(first_zero_index)

	phi_mult : int = 1
	joint_offset_mat = get_rotation_and_translation_matrix(0, joint_offset, axis_name="z")

	for i, phi in enumerate(list(Phi)):

		Li : float = lengths[i]

		print(f"Angle is {phi}")
		ic(i)
		ic(Li)
		# print(f"Generating rotation matrix for part {i}")

		# if i > 1 and i < len(Phi)
		# 	end_effector_loc += np.array([2*r1, 0, 0])

		# get current Li plus some offset due to the joint

		neutral_Li_vec : np.array = np.array([Li, 0, 0])


		# multiply the Li vector into the last matrix
		# current_transform = get_rotation_and_translation_matrix(-1 * phi, neutral_Li_vec, axis_name="z")
		current_transform = get_rotation_and_translation_matrix(0, neutral_Li_vec, axis_name="z")
		# need to add the offset of the bottom half of the joint to the current_transform matrix

		if i == 0:
			pre_offset_mat = get_rotation_and_translation_matrix(phi_mult * phi, joint_offset, axis_name="z")
		elif i < first_zero_index:
			pre_offset_mat = joint_offset_mat
		else:
			pre_offset_mat = np.eye(4)

		type : SEGMENT_TYPE = SEGMENT_TYPE.UPPER_JOINT
		if i == 0:
			type = SEGMENT_TYPE.LOWER_JOINT

		segment_mats.append((type,pre_offset_mat))
		if i == 0:
			segment_mats.append((SEGMENT_TYPE.UPPER_JOINT,joint_offset_mat))

		answers.append(np.eye(4))

		segment_mats.append((SEGMENT_TYPE.ARM,current_transform))

		post_offset_mat = np.eye(4)
		if i < first_zero_index - 1 and i < len(Phi) - 1:
			# post_offset_mat[0:3, -1] = joint_offset
			next_angle : float = Phi[i+1]
			ic(next_angle)
			post_offset_mat = get_rotation_and_translation_matrix(phi_mult * next_angle, joint_offset, axis_name="z" )
		print("Adding post offset matrix")
		ic(post_offset_mat)

	
		type = SEGMENT_TYPE.LOWER_JOINT
		if i >= first_zero_index - 1:
			type = SEGMENT_TYPE.NONE

		segment_mats.append((type,post_offset_mat))
		# ic(cumulative_transform)
		# ic(end_effector)

	ic(segment_mats)

	_ , e = get_end_effector(r1,segment_mats, to_print=True)

	cum_mat = np.eye(4)
	print(f"Final position is {e}")

	print(f"DRAWING ARM!")

	cum_mat = np.eye(4)

	rev_mats = list(segment_mats.copy())

	# transforms = transforms[::-1]
	last_joint_coords = [np.zeros(0), np.zeros(0)]
	cum_mats : List[np.array] = []
	arm_counter = 0
	for i, (type, mat) in enumerate(rev_mats):
		print("------------------------------")
		cum_mat = cum_mat @ mat
		ic(i)
		ic((type, mat))
		ic(cum_mat)
		print(f"Multiplying by ")
			# print(f"Cumulative is ")
			# ic(cumulative_transform)

		sphere_coords = np.array([r1,0,0])
		calc_sphere_coords = np.expand_dims(sphere_coords,axis=1)

		if type == SEGMENT_TYPE.UPPER_JOINT and i != len(rev_mats) - first_zero_index:
			print("Appending joint...")
			print("Getting arm matrix")
			ic(rev_mats[i+1])
			height = rev_mats[i+1][1][0, -1]
			cylinder = Cylinder(height=height, pos=(((height)/2 + r1),0,0), r=r1, alpha=0.8, axis=(1,0,0), c=colors[arm_counter])
			sphere = Sphere(pos=sphere_coords/2, r=r1, alpha=0.8)
			combined = cylinder + sphere
			frames.append(combined.apply_transform(cum_mats[-1]))
			arm_counter+=1
			# frames.append(Sphere(pos=middle_coord, r=r1, alpha=0.8, c="purple").apply_transform(cum_mat))

		if type == SEGMENT_TYPE.LOWER_JOINT:
			print("Appending joint...")
			# frames.append(Sphere(pos=sphere_coords, r=r1, alpha=0.8).apply_transform(cum_mat))
		elif type == SEGMENT_TYPE.ARM:
			print("Appending arm...")
			# frames.append().apply_transform(cum_mat))
		cum_mats.append(cum_mat)

	last_transform = cum_mat
	ic(last_transform)

	# vp.show(frames, axes, interactive=False, viewup=(1,0,0))
	# vp.screenshot(f"./output/output_{index}.png")
	# vp.close()
	assert len(answers) == 4

	answers.append(e)

	# # Function implementation goes here
	return tuple(answers)

def assert_3():
	print("##############################")
	print("  Assertion 3")
	print("##############################")

	# Lentghs of the parts
	L1, L2, L3, L4 = [5, 8, 3, 0]
	Phi = np.array([-30, 50, 30, 0])
	T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)
	
	actual = e
	expected = np.array([18.47772028,  4.71432837,  0. ])

	print(f"{expected=}, {actual=}")
	assert np.allclose(expected, actual)

def assert_1():
	print("##############################")
	print("  Assertion 1")
	print("##############################")
		
	# Lentghs of the parts
	L1, L2, L3, L4 = [5, 8, 3, 0]
	Phi = np.array([30, -50, -30, 0])
	T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4)
	
	actual = e
	expected = np.array([18.47772028, -0.71432837,  0. ])

	print(f"{expected=}, {actual=}")
	assert np.allclose(expected, actual)

def assert_2():
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


if __name__ == '__main__':

	argparser = ArgumentParser()
	argparser.add_argument("--debug", action="store_true")

	args = argparser.parse_args()

	PORT : int = 5678
	if (args.debug):
		print(f"Waiting for client on port {PORT}")
		import debugpy
		debugpy.listen(PORT)
		debugpy.wait_for_client()

	
	assert_1()
	# assert_2()
	# assert_3 ()

	L1, L2, L3, L4 = [5, 8, 3, 0]

	Phi1 = np.linspace(-20, 30, 50)
	Phi2 = np.linspace(-70, -50, 50)
	Phi3 = np.linspace(-20, -30, 50)
	Phi4 = np.linspace(0, 0, 50)
	thing = np.array([Phi1, Phi2, Phi3, Phi4])
	ic(thing)

	for i, pair in enumerate(zip(thing[0], thing[1], thing[2], thing[3])):
		ic(pair)
		Phi = np.array([pair[0], pair[1], pair[2], pair[3]])
		ic(Phi)
		T_01, T_02, T_03, T_04, e = forward_kinematics(Phi, L1, L2, L3, L4, index=i)
	
	# actual = e
	# expected = np.array([18.47772028, -0.71432837,  0. ])