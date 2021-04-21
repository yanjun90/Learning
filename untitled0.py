#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:15:33 2020

@author: yanjunlyu
"""

import os as os
import numpy as np
import shutil as shutil
import csv
import random

import torch

A = torch.Tensor([[1, 1, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1]])

B = torch.Tensor([[1, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 1]])

print(B)

out_degree = torch.sum(B, dim=0)
print (out_degree)
in_degree = torch.sum(B, dim=1)
print (in_degree)

identity = torch.eye(B.size()[0])
print (torch.diagonal(identity))
diag = torch.diagflat(out_degree)
print (diag)

degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A))

###################################################
a = torch.tensor([[1, 2], [2, 3]])
print(a)
b = torch.tensor([[[1,2],[2,3]],[[-1,-2],[-2,-3]]])
print(b)
a.expand(b.size())
print(a)

####################################################
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy

# load a vtk file as input
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("my_input_data.vtk")
reader.Update()

#Grab a scalar from the vtk file
my_vtk_array = reader.GetOutput().GetPointData().GetArray("my_scalar_name")

#Get the coordinates of the nodes and the scalar values
nodes_nummpy_array = vtk_to_numpy(nodes_vtk_array)
my_numpy_array = vtk_to_numpy(my_vtk_array )

x,y,z= nodes_nummpy_array[:,0], nodes_nummpy_array[:,1], nodes_nummpy_array[:,2]








