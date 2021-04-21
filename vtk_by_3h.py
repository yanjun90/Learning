#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:12:29 2020

@author: yanjunlyu
"""

import os
import numpy
import pandas as pd
import csv
import vtk
import re
from vtk.util.numpy_support import vtk_to_numpy


def path_parser(root):
    files_list = os.listdir(root)
    p_list = []
    for n in files_list:
        m = re.match(r'^([0-9]{5,7})\_lh.csv$', n)
        if m:
            p_list.append(m.group(1), m.group(0))
    #p_list.sort()
    return p_list


def read_vtk_file(vtk_file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    Header = reader.GetHeader()

    polydata = reader.GetOutput()

    nCells = polydata.GetNumberOfCells()
    nPolys = polydata.GetNumberOfPolys()
    nLines = polydata.GetNumberOfLines()
    nStrips = polydata.GetNumberOfStrips()
    nPieces = polydata.GetNumberOfPieces()
    nVerts = polydata.GetNumberOfVerts()
    nPoints = polydata.GetNumberOfPoints()
    Points = polydata.GetPoints()
    Point = polydata.GetPoints().GetPoint(0)

    return polydata

def get_points(vtk_file):
    polydata = read_vtk_file(vtk_file)
    points = polydata.GetPoints()
    array = points.GetData()
    numpy_nodes = vtk_to_numpy(array)
    mins = numpy_nodes.min(axis=0) - 1
    
    return numpy_nodes, mins


####################################################################

class TH_group(object):
    def __init__(self, Points, cent_id, travel_1, travel_2, travel_3):
        alist = []
        alist.append(cent_id)
        self.l1 = ' '.split(travel_1.strip())
        self.l2 = ' '.split(travel_1.strip())
        self.l3 = ' '.split(travel_1.strip())
        alist = alist + self.l1 + self.l2 + self.l3
        self.cent_id  = cent_id
        self.true_points = [int(a) for a in alist]
        
        self.lines = []
        self.all_points = numpy.zeros(Points.shape())
        self.Points = Points
        
    def check(tlist, cent_id):
        if len(tlist)%2 == 0:
            r = [numpy.array(tlist[i:i+2]) for i in range(0,len(tlist), 2)]
        else:
            tlist.insert(0, cent_id)
            r = [numpy.array(tlist[i:i+2]) for i in range(0,len(tlist), 2)]
        return r
    
    def gen_points(self):
        [rows, cols] = self.Points.shape()
        for i in self.true_points:
            self.all_points[i,:] = self.all_points[i,:] + self.Points[i,:]
        
    def gen_lines(self):
        self.lines = []
        self.lines.extend(self.check(self.l1, self.cent_id))
        self.lines.extend(self.check(self.l2, self.cent_id))
        self.lines.extend(self.check(self.l3, self.cent_id))
    
    def return_points(self):
        return self.all_points
        
    def return_lines(self):
        return self.lines()
    
    def return_id(self):
        return (str(self.cent_id))
    

#NumOfPoints = polydata.GetNumberOfPoints()

def parse_csv(csv_file, Points):
    all_list = []
    df = pd.read_csv(csv_file, sep=",")
    for index, row in df.iterrows():
        cent_id = row.iloc[0]
        travel_1 = row.iloc[5]
        travel_2 = row.iloc[6]
        travel_3 = row.iloc[7]
        group = TH_group(Points, cent_id, travel_1, travel_2, travel_3)
        group.gen_points()
        group.gen_lines()
        all_list.append(group)
    return all_list

def main(root_path, surfvtk_path, save_path):
    
    # each v is a subject
    v_list = path_parser(root_path)
    for v in v_list:   
        vtk_file = os.path.join(root_path, v[0], surfvtk_path)
        Points, _ = get_points(vtk_file)
        all_list = parse_csv(v[1], Points)
        
        for cent in all_list:
            save_file = os.path.join(save_path, v[0]+cent.return_id(), '.vtk')
            WPoints = vtk.vtkPoints()
            WLines = vtk.vtkCellArray()
            WLine = vtk.vtkLine()
            for i in range(len(cent.return_points())):
                WPoints.InsertNextPoint(cent.return_points()[i,:])
            for i in range(len(cent.return_lines())):
                WLine.GetPointIds().SetId(0, cent.return_lines()[i][0])
                WLine.GetPointIds().SetId(0, cent.return_lines()[i][1])
                WLines.InsertNextPoint(WLine)
                
            Wpolydata = vtk.vtkPolyData()
            Wpolydata.SetPoints(WPoints)
            Wpolydata.SetPolys(WLines)
            Wpolydata.Modified()

            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(Wpolydata)
            writer.SetFileName(save_file)
            writer.Write()
    
    

if __name__ == '__main__':
    root_path = './'
    surfvtk_path = 'Surf/vtk/lh.white.vtk'
    save_path = './h3_vtks'
    main(root_path, surfvtk_path, save_path)
    
    # vtk_file = "./111312/Surf/vtk/lh.white.vtk"
    
    # csv_file = '111211_lh.csv'
    # df = pd.read_csv(csv_file, sep=",")

    # Wpolydata = vtk.vtkPolyData()
    # WTriangle = vtk.vtkTriangle()
    # WTriangle = vtk.vtkLine()
    # WTriangle.GetPointIds().SetId(0, 23)
    # WTriangle.GetPointIds().SetId(1, 43)
    # WTriangle.GetPointIds().SetId(2, 2)
    
    
    