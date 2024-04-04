# 
#  Copyright (C) 2023 Istituto Italiano di Tecnologia (IIT)
#  
#  This software may be modified and distributed under the terms of the
#  GPL-2+ license. See the accompanying LICENSE file for details.
#  
import numpy as np
import time
import threading
import vtk
import vtk.util.numpy_support as vtk_np

class VTKPointCloudOnMesh():
    """
    A class to represent a dynamic point cloud on a mesh.
    
    '''

    Attributes
    ----------
    vtk_points : vtk.vtkPoints
        vtk representation of the point cloud
    vtk_cells : vtkCellArray
        list of the points ids of the point cloud
    poly_data : vtkPolyData
        a geometric structure of vertices
    actor_point_cloud : vtkActor
        the vtk object to display the point cloud in the scene
    actor_mesh : vtkActor
        the vtk object to display the mesh in the scene

    Methods
    -------
    update(thread_lock, update_on, function, args):
        Start a new thread to continuously update the point cloud with a function.
    update_actor(thread_lock, update_on, function, args)
        Update the point cloud depending on the function passed.
    """

    def __init__(self, mesh_path, point_cloud_path, rgb_color=(1, 0, 0), points_size=10.0):
        """
        Constructor.

        Parameters
        ----------
        mesh_path : str
            path to the .obj file of the mesh
        point_cloud_path : str
            path the .txr file of the point cloud
        rgb_color : tuple
            rgb color code of the point cloud
        points_size: float
            size of the points
        """

        # Initialize the point cloud with a 3D point in the origin as default
        nparray = np.loadtxt(point_cloud_path)[:, :3] 
        n_elements = nparray.shape[0]

        # Create the vtk class for the point cloud
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()
        self.poly_data = vtk.vtkPolyData()
        self.vtk_points.SetData(vtk_np.numpy_to_vtk(nparray))
        cells_npy = np.vstack([np.ones(n_elements,dtype=np.int64),
                               np.arange(n_elements,dtype=np.int64)]).T.flatten()
        self.vtk_cells.SetCells(n_elements,vtk_np.numpy_to_vtkIdTypeArray(cells_npy))
        self.poly_data.SetPoints(self.vtk_points)
        self.poly_data.SetVerts(self.vtk_cells)

        # Assing the point cloud to the actor through a mapper
        mapper_point_cloud = vtk.vtkPolyDataMapper()
        mapper_point_cloud.SetInputDataObject(self.poly_data)

        self.actor_point_cloud = vtk.vtkActor()
        self.actor_point_cloud.SetMapper(mapper_point_cloud)
        self.actor_point_cloud.GetProperty().SetRepresentationToPoints()
        self.actor_point_cloud.GetProperty().SetColor(*rgb_color)
        self.actor_point_cloud.GetProperty().SetPointSize(points_size)

        # Initialize the mesh
        reader = vtk.vtkOBJReader()
        reader.SetFileName(mesh_path)
        mapper_mesh = vtk.vtkPolyDataMapper()
        output_port = reader.GetOutputPort()
        mapper_mesh.SetInputConnection(output_port)

        self.actor_mesh = vtk.vtkActor()
        self.actor_mesh.SetMapper(mapper_mesh)


    def update(self, thread_lock, update_on, function, args):
        """
        Start a new thread to continuously update the point cloud with a function.

        Parameters
        ----------
        thread_lock : threading.Lock
            semaphore
        update_on : threading.Event
            signal for the semaphore
        function : python function
            function to update the point cloud
        args : tuple
            arguments of the function passed as argument

        Returns
        -------
        None
        """
        
        thread = threading.Thread(target=self.update_actor, args=(thread_lock, update_on, function, args))
        thread.start()


    def update_actor(self, thread_lock, update_on, function, args):
        """
        Function to update the point cloud

        Parameters
        ----------
        thread_lock : threading.Lock
            semaphore
        update_on : threading.Event
            signal for the semaphore
        function : python function
            function to update the point cloud
        args : tuple
            arguments of the function passed as argument

        Returns
        -------
        None
        """

        while (update_on.is_set()):
            time.sleep(0.01)
            thread_lock.acquire()
            
            # Update the 
            nparray = function(*args)

            # Update the vtk class of the point cloud
            n_elements = nparray.shape[0]
            self.vtk_points.SetData(vtk_np.numpy_to_vtk(nparray))
            cells_npy = np.vstack([np.ones(n_elements,dtype=np.int64),
                                np.arange(n_elements,dtype=np.int64)]).T.flatten()
            self.vtk_cells.SetCells(n_elements,vtk_np.numpy_to_vtkIdTypeArray(cells_npy))
            self.poly_data.SetPoints(self.vtk_points)
            self.poly_data.SetVerts(self.vtk_cells)

            self.poly_data.Modified()

            ### Modify trial
            # pos1 = self.actor_mesh.GetPosition()
            # self.actor_mesh.SetPosition(pos1[0] + 0.01, pos1[1], pos1[2])


            thread_lock.release()


class VTKVisualisation():
    """
    A class to start a vtk visualizer
    
    '''

    Attributes
    ----------
    thread_lock : threading.Lock
        semaphore
    ren : vtkRenderer
        renderer
    axes_actor : vtkAxesActor
        the vtk object to display the axis in the scene
    ren_win : vtkRenderWindow
        the vtk object to set a window for the renderer
    int_ren : vtkRenderWindowInteractor
        the vtk object to interacti with the window
    style : vtkInteractorStyleTrackballCamera
        the vtk object to decide how interact with the window

    Methods
    -------
    update(thread_lock, update_on, function, args):
        Method to update the render window.

    """

    def __init__(self, thread_lock, actor_wrapper, background=(0, 0, 0), axis=False):
        """
        Constructor.

        Parameters
        ----------
        thread_lock : threading.Lock
            semaphore
        actor_wrapper : VTKPointCloudOnMesh
            wrapper for the actors
        background : tuple
            rgb color code for the window background
        axis : bool
            boolean to decide wheater render the axis in the origin or not
        """
         
        # Assign the semaphore
        self.thread_lock = thread_lock

        # Initialize the renderer
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*background)

        # Add the point cloud
        self.ren.AddActor(actor_wrapper.actor_point_cloud)
        
        # Add the mesh
        self.ren.AddActor(actor_wrapper.actor_mesh)

        # Add the axes
        if axis:
            self.axes_actor = vtk.vtkAxesActor()
            self.axes_actor.AxisLabelsOff()
            self.axes_actor.SetTotalLength(1, 1, 1)
            self.ren.AddActor(self.axes_actor)

        # Create a window
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.AddRenderer(self.ren)

        # Create an interactor
        self.int_ren = vtk.vtkRenderWindowInteractor()
        self.int_ren.SetRenderWindow(self.ren_win)
        self.int_ren.Initialize()

        # Change the style to actively interact with the camera in the scene
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.int_ren.SetInteractorStyle(self.style)

        # Add a function to the loop
        self.int_ren.AddObserver("TimerEvent", self.update_visualisation)
        dt = 30 # ms
        timer_id = self.int_ren.CreateRepeatingTimer(dt)


    def update_visualisation(self, obj=None, event=None):
        """
        Method to update the render window.

        Parameters
        ----------
        obj : vtkXRenderWindowInteractor, optional
            convenience object that provides event bindinds to common graphics functions
        event : str, optional
            name of the event to handle
        """

        time.sleep(0.01)
        self.thread_lock.acquire()
        self.ren.GetRenderWindow().Render()
        self.thread_lock.release()
