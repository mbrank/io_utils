import math
import numpy as np
import copy
import os
import shutil
import pickle
import h5py


# ---------------------------------------------------------------------------
# Pickle helpers
# ---------------------------------------------------------------------------
def save_object(obj, filename):
    """Save a Python object to ``filename`` using pickle protocol 2."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=2)


def load_object(filename):
    """Load a previously pickled Python object from ``filename``."""
    with open(filename, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
def cart_to_cyl(x, y, z):
    """Convert Cartesian (x, y, z) to cylindrical (r, theta, z); theta in radians."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta, z


def cyl_to_cart(r, theta, z):
    """Convert cylindrical (r, theta, z) to Cartesian (x, y, z); theta in radians."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def check_clockwise_order(nodes, cells):
    """
    Determines if cells (quadrilaterals) have clockwise or counter-clockwise orientation.
    
    Args:
        nodes: Array of node coordinates
        cells: Array of cell definitions (indices to nodes array)
        
    Returns:
        sign: Array with values -1 for counter-clockwise or 1 for clockwise orientation
    """
    nodes = np.array(nodes).astype('float')
    cells = np.array(cells).astype('int')

    # Get the coordinates of the four corners of each cell
    a = nodes[cells[:, 0]]
    b = nodes[cells[:, 1]]
    c = nodes[cells[:, 2]]
    d = nodes[cells[:, 3]]

    # Check counter-clockwise orientation using shoelace formula components
    a2b = (b[:, 0]-a[:, 0])*(b[:, 1]+a[:, 1])
    b2c = (c[:, 0]-b[:, 0])*(c[:, 1]+b[:, 1])
    c2d = (d[:, 0]-c[:, 0])*(d[:, 1]+c[:, 1])
    d2a = (a[:, 0]-d[:, 0])*(a[:, 1]+d[:, 1])
    sum_cc = a2b+b2c+c2d+d2a
    sign = np.zeros(cells.shape[0])
    for _sum in enumerate(sum_cc):
        _sum_id = _sum[0]
        _sum = _sum[1]
        if _sum > 0:
            sign[_sum_id] = -1
        else:
            sign[_sum_id] = 1

    return sign


def convert_quad_to_tria(cells):
    """
    Converts quadrilateral cells to triangular cells by splitting each quad into two triangles.
    
    Args:
        cells: Array of quadrilateral cell definitions
        
    Returns:
        cells_tria: Array of triangular cell definitions
    """
    # Cells have indices in clockwise/counterclocwise order
    cells_tria = []
    for cell in cells:
        cells_tria.append([cell[0], cell[1], cell[2]])
        cells_tria.append([cell[0], cell[2], cell[3]])

    return cells_tria


def convert_quaddata_to_triadata(_data):
    """
    Converts data associated with quadrilateral cells to corresponding triangular cell data.
    
    Args:
        _data: Dictionary containing cell data and other mesh information
        
    Returns:
        data_cells_tria: Dictionary with data converted to triangular cells
    """
    cells = _data["cells"]
    data_cells_tria = copy.deepcopy(_data)
    cells_tria = convert_quad2tria(cells)
    for name in data_cells_tria:
        data = data_cells_tria[name]
        if len(data) == len(cells):
            data_tria = []
            for value in data:
                data_tria.append(value)
                data_tria.append(value)

            data_cells_tria[name] = data_tria

    data_cells_tria["cells"] = cells_tria
    return data_cells_tria


def convert_node_data_to_cell_data(nodes, cells, node_data):
    """
    Converts data associated with nodes to cell data by averaging node values for each cell.
    
    Args:
        nodes: Array of node coordinates
        cells: Array of cell definitions
        node_data: Array of data values associated with nodes
        
    Returns:
        cell_data: Array of data values associated with cells
    """
    nodes = np.array(nodes).astype('float')
    cells = np.array(cells).astype('int')
    cell_data = np.zeros(cells.shape[0])
    for cell in enumerate(cells):
        cell_id = cell[0]
        cell = cell[1]
        q_avg = np.sum(np.fromiter([node_data[cell[i]] for i in range(cell.shape[0])],
                                   dtype='float'))/cells.shape[1]
        cell_data[cell_id] = q_avg

    return cell_data


def cell_centroid(nodes, cells):
    """
    Calculates centroids for each cell in a mesh.
    
    Args:
        nodes: Array of node coordinates
        cells: Array of cell definitions
        
    Returns:
        [Cx, Cy]: Arrays of x and y coordinates of cell centroids
    """
    # Returns array of centroids of quadrangles
    # Input array of nodes and array of quadrangle indices
    # The cell must lie in x, y plane
    nodes = np.array(nodes).astype('float')
    cells = np.array(cells).astype('int')
    
    # Handle triangular cells
    if cells.shape[1] == 3:
        centroids_x = []
        centroids_y = []
        for cell in cells:
            x0 = nodes[cell[0]][0]
            y0 = nodes[cell[0]][1]
            x1 = nodes[cell[1]][0]
            y1 = nodes[cell[1]][1]
            x2 = nodes[cell[2]][0]
            y2 = nodes[cell[1]][1]
            centroid = calc_barycenter(x0, y0, x1, y1, x2, y2)
            centroids_x.append(centroid[0])
            centroids_y.append(centroid[1])
        return [np.array(centroids_x),
                np.array(centroids_y)]

    # Handle quadrilateral cells
    a = nodes[cells[:, 0]]
    b = nodes[cells[:, 1]]
    c = nodes[cells[:, 2]]
    d = nodes[cells[:, 3]]

    # Calculate centroid using shoelace formula components
    sm0_x = ((a[:, 0]+b[:, 0])*(a[:, 0]*b[:, 1]-b[:, 0]*a[:, 1]))
    sm1_x = ((b[:, 0]+c[:, 0])*(b[:, 0]*c[:, 1]-c[:, 0]*b[:, 1]))
    sm2_x = ((c[:, 0]+d[:, 0])*(c[:, 0]*d[:, 1]-d[:, 0]*c[:, 1]))
    sm3_x = ((d[:, 0]+a[:, 0])*(d[:, 0]*a[:, 1]-a[:, 0]*d[:, 1]))
    
    sm0_y = ((a[:, 1]+b[:, 1])*(a[:, 0]*b[:, 1]-b[:, 0]*a[:, 1]))
    sm1_y = ((b[:, 1]+c[:, 1])*(b[:, 0]*c[:, 1]-c[:, 0]*b[:, 1]))
    sm2_y = ((c[:, 1]+d[:, 1])*(c[:, 0]*d[:, 1]-d[:, 0]*c[:, 1]))
    sm3_y = ((d[:, 1]+a[:, 1])*(d[:, 0]*a[:, 1]-a[:, 0]*d[:, 1]))

    area = cells_area(nodes, cells)
    sign = check_clockwise_order(nodes, cells)
    Cx = 1/(6*area)*(sm0_x+sm1_x+sm2_x+sm3_x)*sign
    Cy = 1/(6*area)*(sm0_y+sm1_y+sm2_y+sm3_y)*sign
    return [Cx, Cy]


def calc_quadrangle_centroid_shoelace_area(pt_x, pt_y):
    """
    Calculates the centroid of a quadrilateral using the shoelace formula.
    
    Args:
        pt_x: Array of x-coordinates of the quad vertices
        pt_y: Array of y-coordinates of the quad vertices
        
    Returns:
        [Cx, Cy]: Coordinates of the centroid
    """
    a = [pt_x[0], pt_y[0]]
    b = [pt_x[1], pt_y[1]]
    c = [pt_x[2], pt_y[2]]
    d = [pt_x[3], pt_y[3]]

    # Calculate centroid using shoelace formula components
    sm0_x = ((a[0]+b[0])*(a[0]*b[1]-b[0]*a[1]))
    sm1_x = ((b[0]+c[0])*(b[0]*c[1]-c[0]*b[1]))
    sm2_x = ((c[0]+d[0])*(c[0]*d[1]-d[0]*c[1]))
    sm3_x = ((d[0]+a[0])*(d[0]*a[1]-a[0]*d[1]))
    
    sm0_y = ((a[1]+b[1])*(a[0]*b[1]-b[0]*a[1]))
    sm1_y = ((b[1]+c[1])*(b[0]*c[1]-c[0]*b[1]))
    sm2_y = ((c[1]+d[1])*(c[0]*d[1]-d[0]*c[1]))
    sm3_y = ((d[1]+a[1])*(d[0]*a[1]-a[0]*d[1]))

    area = shoelace_area(pt_x, pt_y)
    Cx = 1/(6*area)*(sm0_x+sm1_x+sm2_x+sm3_x)
    Cy = 1/(6*area)*(sm0_y+sm1_y+sm2_y+sm3_y)
    return [Cx, Cy]


def calc_cells_area(nodes, cells):
    """
    Calculates the area of each cell in a mesh using the shoelace formula.
    
    Args:
        nodes: Array of node coordinates
        cells: Array of cell definitions
        
    Returns:
        area: Array of cell areas
    """
    nodes = np.array(nodes).astype('float')
    cells = np.array(cells).astype('int')

    # Handle triangular cells
    if cells.shape[1] == 3:
        a = nodes[cells[:, 0]]
        b = nodes[cells[:, 1]]
        c = nodes[cells[:, 2]]
        areas = calc_tria_area(a[:, 0], a[:, 1],
                                    b[:, 0], b[:, 1],
                                    c[:, 0], c[:, 1])
        return areas

    # Handle quadrilateral cells
    a = nodes[cells[:, 0]]
    b = nodes[cells[:, 1]]
    c = nodes[cells[:, 2]]
    d = nodes[cells[:, 3]]

    area = np.zeros(len(cells))
    for cell in enumerate(cells):
        cell_ind = cell[0]
        cell = cell[1]
        x_list = [nodes[cell[0]][0],
                  nodes[cell[1]][0],
                  nodes[cell[2]][0],
                  nodes[cell[3]][0]]
        y_list = [nodes[cell[0]][1],
                  nodes[cell[1]][1],
                  nodes[cell[2]][1],
                  nodes[cell[3]][1]]
        a1, a2 = 0, 0
        x_list.append(x_list[0])  # Close the polygon
        y_list.append(y_list[0])
        # Apply shoelace formula
        for j in range(len(x_list)-1):
            a1 += x_list[j]*y_list[j+1]
            a2 += y_list[j]*x_list[j+1]
        l = abs(a1-a2)/2
        area[cell_ind] = l

    return np.array(area)


def calc_shoelace_area(x_list, y_list):
    """
    Calculates the area of a polygon using the shoelace formula.
    
    Args:
        x_list: List of x-coordinates of polygon vertices
        y_list: List of y-coordinates of polygon vertices
        
    Returns:
        l: Area of the polygon
    """
    a1, a2 = 0, 0
    x_list.append(x_list[0])  # Close the polygon
    y_list.append(y_list[0])
    # Apply shoelace formula
    for j in range(len(x_list)-1):
        a1 += x_list[j]*y_list[j+1]
        a2 += y_list[j]*x_list[j+1]
    l = abs(a1-a2)/2
    return l


def calc_node_distance(pt0, pt1):
    """
    Calculates the Euclidean distance between two points in 2D.
    
    Args:
        pt0: Coordinates of the first point [x,y]
        pt1: Coordinates of the second point [x,y]
        
    Returns:
        distance: Euclidean distance between points
    """
    return math.sqrt((pt0[0]-pt1[0])**2+(pt0[1]-pt1[1])**2)


def calc_max_quad_area(a, b, c, d):
    """
    Calculates the maximum possible area of a quadrilateral using Brahmagupta's formula.
    
    Args:
        a, b, c, d: Side lengths of the quadrilateral
        
    Returns:
        max_area: Maximum possible area of the quadrilateral
    """
    # Calculating the semi-perimeter of the given quadrilateral
    semiperimeter = (a + b + c + d) / 2

    # Applying Brahmagupta's formula to get maximum area of quadrilateral
    return math.sqrt((semiperimeter - a) *
                     (semiperimeter - b) *
                     (semiperimeter - c) *
                     (semiperimeter - d))


def calc_barycenter2D(x1, y1, x2, y2, x3, y3):
    """
    Calculates the barycenter (centroid) of a triangle in 2D.
    
    Args:
        x1, y1, x2, y2, x3, y3: Coordinates of the triangle vertices
        
    Returns:
        [bx, by]: Coordinates of the barycenter
    """
    return [(x1+x2+x3)/3, (y1+y2+y3)/3]


def calc_barycenter3D(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """
    Calculates the barycenter (centroid) of a triangle in 3D.
    
    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3: Coordinates of the triangle vertices
        
    Returns:
        [bx, by, bz]: Coordinates of the barycenter
    """
    return [(x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3]


def calc_cells_barycenter3D(nodes, cells):
    """
    Calculates the barycenters of triangular cells in a 3D mesh.
    
    Args:
        nodes: Array of node coordinates
        cells: Array of cell definitions
        
    Returns:
        cell_barycenters: Coordinates of the cell barycenters
    """
    nodes = np.array(nodes).astype('float')
    cells = np.array(cells).astype('int')

    # Check if cells are triangles
    if cells.shape[1] == 3:
        a = nodes[cells[:, 0]]
        b = nodes[cells[:, 1]]
        c = nodes[cells[:, 2]]
        cell_barycenters = calc_barycenter3D(a[:, 0], a[:, 1], a[:, 2],
                                                  b[:, 0], b[:, 1], b[:, 2],
                                                  c[:, 0], c[:, 1], c[:, 2])
        return cell_barycenters
    else:
        print("Cells are not triangles in generate_ggd_vtks.py ->"
              "calc_cell_barycenter3D()")


def calc_tria_area(x1, y1, x2, y2, x3, y3):
    """
    Calculates the area of a triangle in 2D using the shoelace formula.
    
    Args:
        x1, y1, x2, y2, x3, y3: Coordinates of the triangle vertices
        
    Returns:
        area: Area of the triangle
    """
    return np.abs(x1*y2+x2*y3+x3*y1-y1*x2-y2*x3-y3*x1)/2


def find_centroid_of_4_coordinates(a, b, c, d):
    """
    Finds the centroid of a quadrilateral by splitting it into two triangles and computing their centroids.
    
    Args:
        a, b, c, d: 3D coordinates of the four vertices
        
    Returns:
        xc, yc, zc: Coordinates of the centroid
    """
    # Find the centroid of triangle 1
    x1 = (a[0]+b[0]+d[0])/3
    y1 = (a[1]+b[1]+d[1])/3
    z1 = (a[2]+b[2]+d[2])/3
    # Find the centroid of triangle 2
    x2 = (b[0]+c[0]+d[0])/3
    y2 = (b[1]+c[1]+d[1])/3
    z2 = (b[2]+c[2]+d[2])/3
    # Find the midpoint of line between the centroids of triangle 1 and 2
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    zc = (z1+z2)/2
    return xc, yc, zc


def generate_tetra_vtk(file_name, vtk_data):
    """
    Generates a VTK file for tetrahedral mesh elements.
    
    Args:
        file_name: Name of the output VTK file
        vtk_data: Dictionary containing mesh data
    """
    vtk_data = copy.deepcopy(vtk_data)

    nodes = vtk_data["nodes"]
    faces = vtk_data["cells"]
    vtk_data.pop("nodes")
    vtk_data.pop("cells")
    node_data = {}
    face_data = {}
    for name in vtk_data:
        if "_nodes" in name:
            node_data[name] = vtk_data[name]
        else:
            face_data[name] = vtk_data[name]

    n_nodes = len(nodes)
    n_faces = len(faces)
    face_type = 5
    if len(faces[0]) == 4:
        face_type = 10  # Tetrahedron
    elif len(faces[0]) == 8:
        face_type = 12  # Hexahedron
        
    # Write VTK file in ASCII format
    with open(file_name, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mesh_1\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS "+str(n_nodes)+" float\n")
        for node in nodes:
            f.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")

        f.write("\n")
        f.write("CELLS "+str(n_faces)+" "+str(n_faces*(len(faces[0])+1))+"\n")

        faces = np.array(faces, dtype='int')
        if face_type == 10:
            for face in faces:
                f.write("4 " +
                        str(face[0])+" " +
                        str(face[1])+" " +
                        str(face[2])+" " +
                        str(face[3])+"\n")
        elif face_type == 12:
            for face in faces:
                f.write("8 " +
                        str(face[0])+" " +
                        str(face[1])+" " +
                        str(face[2])+" " +
                        str(face[3])+" " +
                        str(face[4])+" " +
                        str(face[5])+" " +
                        str(face[6])+" " +
                        str(face[7])+"\n")

        f.write("\n")
        f.write("CELL_TYPES "+str(n_faces)+"\n")
        for i in range(n_faces):
            f.write(str(face_type)+"\n")

        # Write node data
        for data in enumerate(node_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("POINT_DATA "+str(n_nodes)+"\n")

            f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in node_data[val_name]:
                f.write("  "+str(value)+"\n")

        # Write face data
        for data in enumerate(face_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("CELL_DATA "+str(n_faces)+"\n")

            if len(np.array(face_data[val_name]).shape) == 2:
                f.write("VECTORS "+val_name.replace(" ", "")+" float\n")
                for value in face_data[val_name]:
                    for vl in value:
                        f.write("  "+str(vl))
                    f.write("\n")
            else:
                f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for value in face_data[val_name]:
                    f.write("  "+str(value)+"\n")



def generate_faces_vtk(file_name, vtk_data):
    """
    Generates a VTK file for triangular or quadrilateral face elements.
    
    Args:
        file_name: Name of the output VTK file
        vtk_data: Dictionary containing mesh data
    """
    vtk_data = copy.deepcopy(vtk_data)

    nodes = vtk_data["nodes"]
    faces = vtk_data["cells"]
    vtk_data.pop("nodes")
    vtk_data.pop("cells")
    node_data = {}
    face_data = {}
    for name in vtk_data:
        if "_nodes" in name:
            node_data[name] = vtk_data[name]
        else:
            face_data[name] = vtk_data[name]

    n_nodes = len(nodes)
    n_faces = len(faces)
    face_type = 5
    if len(faces[0]) == 3:
        face_type = 5  # Triangle
    elif len(faces[0]) == 4:
        face_type = 9  # Quad
        
    # Write VTK file in ASCII format
    with open(file_name, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mesh_1\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS "+str(n_nodes)+" float\n")
        for node in nodes:
            f.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")

        f.write("\n")
        f.write("CELLS "+str(n_faces)+" "+str(n_faces*(len(faces[0])+1))+"\n")

        faces = np.array(faces, dtype='int')
        if face_type == 5:
            for face in faces:
                f.write("3 " +
                        str(face[0])+" " +
                        str(face[1])+" " +
                        str(face[2])+"\n")
        elif face_type == 9:
            for face in faces:
                f.write("4 " +
                        str(face[0])+" " +
                        str(face[1])+" " +
                        str(face[2])+" " +
                        str(face[3])+"\n")

        f.write("\n")
        f.write("CELL_TYPES "+str(n_faces)+"\n")
        for i in range(n_faces):
            f.write(str(face_type)+"\n")

        # Write node data
        for data in enumerate(node_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("POINT_DATA "+str(n_nodes)+"\n")

            f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in node_data[val_name]:
                f.write("  "+str(value)+"\n")

        # Write face data
        for data in enumerate(face_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("CELL_DATA "+str(n_faces)+"\n")

            if len(np.array(face_data[val_name]).shape) == 2:
                f.write("VECTORS "+val_name.replace(" ", "")+" float\n")
                for value in face_data[val_name]:
                    for vl in value:
                        f.write("  "+str(vl))
                    f.write("\n")
            else:
                f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for value in face_data[val_name]:
                    f.write("  "+str(value)+"\n")


def generate_faces_h5(h5_path,
                      data,
                      mesh_group="/mesh",
                      fields_group="/fields",
                      compression="gzip",
                      compression_opts=4,
                      overwrite_datasets=True,
                      ):
    """Write a triangular mesh + per-cell / per-node fields to an HDF5 file.

    Layout:
      ``/mesh/nodes``           (n,3) float
      ``/mesh/cells``           (m,3) int (triangles)
      ``/fields/cells/<name>``  (m,)  field arrays on cells (keys ending in ``_cells``)
      ``/fields/nodes/<name>``  (n,)  field arrays on nodes (keys ending in ``_nodes``)
      ``/fields/misc/<name>``   any other 1D arrays the caller supplies
    """
    if "nodes" not in data or "cells" not in data:
        raise KeyError('Expected at least "nodes" and "cells" in data.')

    nodes = np.asarray(data["nodes"])
    cells = np.asarray(data["cells"])

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f'"nodes" must be (n,3); got {nodes.shape}')
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f'"cells" must be (m,3); got {cells.shape}')

    nodes = nodes.astype(np.float32, copy=False)
    if np.issubdtype(cells.dtype, np.floating) and np.allclose(cells, np.round(cells)):
        cells = np.round(cells).astype(np.int32)
    else:
        cells = cells.astype(np.int32, copy=False)

    n = nodes.shape[0]
    m = cells.shape[0]

    nodes_chunks = (min(8192, n), 3)
    cells_chunks = (min(16384, m), 3)
    field_chunks_cells = (min(16384, m),)
    field_chunks_nodes = (min(8192, n),)

    def _create_or_replace(grp, name, arr, chunks):
        if overwrite_datasets and name in grp:
            del grp[name]
        grp.create_dataset(
            name,
            data=arr,
            dtype=arr.dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True if compression else False,
        )

    with h5py.File(h5_path, "a") as h5:
        g_mesh = h5.require_group(mesh_group)
        g_fields = h5.require_group(fields_group)
        g_cell_fields = g_fields.require_group("cells")
        g_node_fields = g_fields.require_group("nodes")

        g_mesh.attrs["format"] = "mesh_fields_v1"
        g_mesh.attrs["nodes_shape"] = nodes.shape
        g_mesh.attrs["cells_shape"] = cells.shape

        _create_or_replace(g_mesh, "nodes", nodes, nodes_chunks)
        _create_or_replace(g_mesh, "cells", cells, cells_chunks)

        for k, v in data.items():
            if k in ("nodes", "cells"):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.reshape(-1)
            if arr.ndim != 1:
                raise ValueError(f'Field "{k}" must be 1D (or (N,1)); got shape {arr.shape}')

            if k.endswith("_cells"):
                if arr.shape[0] != m:
                    raise ValueError(
                        f'Field "{k}" length {arr.shape[0]} does not match n_cells={m}'
                    )
                _create_or_replace(g_cell_fields, k[:-6], arr, field_chunks_cells)
            elif k.endswith("_nodes"):
                if arr.shape[0] != n:
                    raise ValueError(
                        f'Field "{k}" length {arr.shape[0]} does not match n_nodes={n}'
                    )
                _create_or_replace(g_node_fields, k[:-6], arr, field_chunks_nodes)
            else:
                g_misc = g_fields.require_group("misc")
                _create_or_replace(g_misc, k, arr, (min(16384, arr.shape[0]),))


def generate_edges_vtk(file_name, vtk_data):
    """
    Generates a VTK file for edge elements.
    
    Args:
        file_name: Name of the output VTK file
        vtk_data: Dictionary containing mesh data
    """
    vtk_data = copy.deepcopy(vtk_data)

    nodes = np.array(vtk_data["nodes"]).astype('float')
    edges = np.array(vtk_data["cells"]).astype('int')
    vtk_data.pop("nodes")
    vtk_data.pop("cells")
    node_data = {}
    edge_data = {}
    for name in vtk_data:
        if "_nodes" in name:
            node_data[name] = vtk_data[name]
        else:
            edge_data[name] = vtk_data[name]

    n_nodes = len(nodes)
    n_edges = len(edges)
    edge_type = 3  # Line cell type in VTK

    # Write VTK file in ASCII format
    with open(file_name, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mesh_1\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS "+str(n_nodes)+" float\n")
        for node in nodes:
            f.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")

        f.write("\n")
        f.write("CELLS "+str(n_edges)+" "+str(n_edges*(2+1))+"\n")

        for edge in edges:
            f.write("2 " +
                    str(edge[0])+" " +
                    str(edge[1])+"\n")

        f.write("CELL_TYPES "+str(n_edges)+"\n")
        for i in range(n_edges):
            f.write("3\n")  # VTK_LINE = 3

        # Write node data
        for data in enumerate(node_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("POINT_DATA "+str(n_nodes)+"\n")

            f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in node_data[val_name]:
                f.write("  "+str(value)+"\n")

        # Write edge data
        for data in enumerate(edge_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            if iterator == 0:
                f.write("CELL_DATA "+str(n_edges)+"\n")

            f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for value in edge_data[val_name]:
                f.write("  "+str(value)+"\n")




def generate_points_vtk(file_name, vtk_data):
    """
    Generates a VTK file containing only point (node) data without any cells.
    
    This function writes the points and their associated scalar data to a VTK file in
    ASCII format. It only includes nodes without any connectivity information.
    
    Args:
        file_name: Name/path of the output VTK file to be created
        vtk_data: Dictionary containing mesh data with 'nodes' and optionally node data fields
                  with names containing '_nodes'
    """
    # Create a deep copy to avoid modifying the original data
    vtk_data = copy.deepcopy(vtk_data)
    
    # Extract the node coordinates
    nodes = vtk_data["nodes"]
    vtk_data.pop("nodes")
    
    # Extract data associated with nodes (fields ending with '_nodes')
    node_data = {}
    for name in vtk_data:
        if "_nodes" in name:
            node_data[name] = vtk_data[name]
    
    n_nodes = len(nodes)
    
    # Write the VTK file in ASCII format
    with open(file_name, "w") as f:
        # Write the VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mesh_1\n")  # Title
        f.write("ASCII\n")   # File format
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write the point coordinates
        f.write("POINTS "+str(n_nodes)+" float\n")
        for node in nodes:
            f.write(str(node[0])+" "+str(node[1])+" "+str(node[2])+"\n")
        
        # Write the point data (scalar fields associated with nodes)
        for data in enumerate(node_data):
            iterator = data[0]
            val_name = data[1]
            f.write("\n")
            
            # Write POINT_DATA header only once before the first data field
            if iterator == 0:
                f.write("POINT_DATA "+str(n_nodes)+"\n")
            
            # Write the scalar field header
            f.write("SCALARS "+val_name.replace(" ", "")+" float 1\n")
            f.write("LOOKUP_TABLE default\n")
            
            # Write the scalar values for each node
            for value in node_data[val_name]:
                f.write("  "+str(value)+"\n")



def read_vtk(file_name):
    """
    Reads a VTK file and converts its contents to a dictionary.
    
    Args:
        file_name: Name of the input VTK file
        
    Returns:
        data: Dictionary containing mesh data extracted from VTK file
    """
    data = {}
    with open(file_name, "r") as f:
        f.readline()  # VTK version
        f.readline()  # Header
        f.readline()  # Format (ASCII/BINARY)
        line = f.readline()  # DATASET type
        if line.isspace():
            f.readline()

        # Read nodes
        n_pts = int(f.readline().split()[1])
        nodes = []
        for node in range(n_pts):
            nodes.append([float(coord) for coord in f.readline().split()])

        nodes = np.array(nodes)
        node0 = nodes[0, :]
        if node0[1] == 0:
            nodes = np.array([nodes[:, 0], nodes[:, 1], nodes[:, 2]])
            nodes = nodes.T.tolist()

        data["nodes"] = nodes

        line = f.readline()
        if line.isspace():
            line = f.readline()

        # Read cells
        n_cells = int(line.split()[1])
        cells = []
        for cell in range(n_cells):
            cells.append([float(cell_id) for cell_id in f.readline().split()[1:]])

        data["cells"] = cells
        line = f.readline()
        if line == '':
            line = f.readline()

        # Skip cell types
        for cell in range(n_cells):
            f.readline()

        point_data = False
        cell_data = False
        scalar_type = None
        vector_type = None

        # Read data fields
        for ln in f:
            line = ln
            if line == '\n':
                line = f.readline()

            if line.split()[0] == "POINT_DATA":
                point_data = True
                cell_data = False
                data_name = f.readline().split()[1]

            elif line.split()[0] == "CELL_DATA":
                point_data = False
                cell_data = True
                line = f.readline()
                if line.split()[0] == 'SCALARS':
                    scalar_type = True
                    vector_type = False
                    f.readline()  # Skip LOOKUP_TABLE line

                elif line.split()[0] == 'VECTORS':
                    scalar_type = False
                    vector_type = True

                data_name = line.split()[1]

            elif line.split()[0] == "SCALARS":
                scalar_type = True
                vector_type = False
                data_name = line.split()[1]
                f.readline()  # Skip LOOKUP_TABLE line

            elif line.split()[0] == "VECTORS":
                scalar_type = False
                vector_type = True
                data_name = line.split()[1]

            # Read point data values
            if point_data:
                data_set = []
                for value in range(n_pts):
                    data_set.append(float(f.readline().split()[0]))

                if "_nodes" in data_name:
                    data[data_name] = data_set
                else:
                    data[data_name+"_nodes"] = data_set
                point_data = True
                cell_data = False

            # Read cell data values
            elif cell_data:
                data_set = []
                i=0
                for value in range(n_cells):
                    if scalar_type:
                        val = float(f.readline().split()[0])
                    elif vector_type:
                        val = [float(i) for i in f.readline().split()]

                    data_set.append(val)
                    i += 1

                if "_cells" in data_name:
                    data[data_name] = data_set
                else:
                    data[data_name+"_cells"] = data_set

                point_data = False
                cell_data = True

    return data



def remove_orphan_nodes(data):
    """
    Removes nodes that are not referenced by any cell from the mesh data.
    
    Args:
        data: Dictionary containing mesh data
        
    Returns:
        data: Updated dictionary with orphan nodes removed
    """
    data = copy.deepcopy(data)
    nodes = np.array(data["nodes"]).astype("float")
    cells = np.array(data["cells"]).astype("int")
    new_cells = np.zeros(cells.shape)
    
    # Find nodes that are actually used by cells
    old_nodes_ids = np.isin(np.arange(nodes.shape[0]), cells).astype('int')
    old_nodes_ids = np.argwhere(old_nodes_ids).T[0]
    new_nodes = nodes[old_nodes_ids]
    
    # Update cell indices to point to new node indices
    for id_nr in enumerate(old_nodes_ids):
        new_cells[np.where(cells==id_nr[1])] = id_nr[0]

    data["nodes"] = new_nodes
    data["cells"] = new_cells
    
    # Update node data fields if present
    for name in data:
        if "_nodes" in name:
            print("Warning: data of "+str(name)+" on orphan nodes is lost!")
            data[name] = data[name][old_nodes_ids]

    return data




def merge_two_triangular_meshes(vtk_data0, vtk_data1):
    """
    Merges two triangular meshes into a single mesh.
    
    Args:
        vtk_data0: Dictionary containing the first mesh data
        vtk_data1: Dictionary containing the second mesh data
        
    Returns:
        new_data: Dictionary containing the merged mesh data
    """
    vtk_data0 = copy.deepcopy(vtk_data0)
    vtk_data1 = copy.deepcopy(vtk_data1)

    # Get nodes and cells from first mesh
    nodes0 = np.array(vtk_data0["nodes"]).astype('float')
    n_nodes0 = nodes0.shape[0]
    cells0 = np.array(vtk_data0["cells"]).astype('int')
    n_cells0 = cells0.shape[0]

    vtk_data0.pop("nodes")
    vtk_data0.pop("cells")

    # Get nodes and cells from second mesh
    nodes1 = np.array(vtk_data1["nodes"]).astype('float')
    n_nodes1 = nodes1.shape[0]
    cells1 = np.array(vtk_data1["cells"]).astype('int')
    n_cells1 = cells1.shape[0]

    vtk_data1.pop("nodes")
    vtk_data1.pop("cells")

    # Concatenate nodes and update cell indices for the second mesh
    new_nodes = np.concatenate((nodes0, nodes1))
    new_cells = np.concatenate((cells0, cells1+n_nodes0))

    new_data = {"nodes": new_nodes, "cells": new_cells}

    # Merge data fields
    for name in vtk_data0:
        if name in vtk_data1:
            new_data[name] = np.concatenate((vtk_data0[name], vtk_data1[name]))
            vtk_data1.pop(name)
        elif "_nodes" in name:
            new_data[name] = np.concatenate((vtk_data0[name], np.zeros(n_nodes1)))
        elif "_cells" in name:
            new_data[name] = np.concatenate((vtk_data0[name], np.zeros(n_cells1)))

    for name in vtk_data1:
        if "_nodes" in name:
            new_data[name] = np.concatenate((np.zeros(n_nodes0), vtk_data1[name]))
        elif "_cells" in name:
            new_data[name] = np.concatenate((np.zeros(n_cells0), vtk_data1[name]))

    return new_data


def remove_double_nodes(data):
    """
    Identifies and removes duplicate nodes from mesh data.
    
    Args:
        data: Dictionary containing mesh data
        
    Returns:
        data: Updated dictionary with duplicate nodes removed
    """
    print("Warning: Data on double nodes is lost")
    data = copy.deepcopy(data)
    nodes = np.array(data["nodes"]).astype('float')
    n_nodes = nodes.shape[0]
    cells = np.array(data["cells"]).astype('int')
    n_cells = cells.shape[0]

    data.pop("nodes")
    data.pop("cells")

    # Find repeating nodes and update cell references
    for nd in enumerate(nodes):
        id_repeating_nodes = np.where(np.all(np.isclose(nd[1], nodes), axis=1)==True)
        if id_repeating_nodes[0].shape[0] == 2:
            cells[cells == id_repeating_nodes[0][1]] = id_repeating_nodes[0][0]

    # Reorder nodes and remove new orphan nodes
    new_cells = np.zeros(cells.shape)
    old_nodes_ids = np.isin(np.arange(nodes.shape[0]), cells).astype('int')
    old_nodes_ids = np.argwhere(old_nodes_ids).T[0]
    new_nodes = nodes[old_nodes_ids]
    data["nodes"] = new_nodes
    
    # Update node data fields
    for name in data:
        if "_nodes" in name:
            data[name] = data[name][old_nodes_ids]
            print("    Data of "+str(name)+" on double nodes is deleted.")

    # Update cell indices
    for id_nr in enumerate(old_nodes_ids):
        new_cells[np.where(cells==id_nr[1])] = id_nr[0]

    data["cells"] = new_cells

    return data


def convert_dat_to_py(dat_file):
    """
    Reads a .dat mesh file and extracts nodes, edges, and triangles.
    
    Args:
        dat_file: Path to the input .dat file
        
    Returns:
        nodes: Array of node coordinates
        edges: Array of edge definitions
        triangles: Array of triangle definitions
    """
    nodes = []
    edges = []
    triangles = []
    
    with open(dat_file, "r") as f:
        # Read number of nodes and cells
        [n_nodes, n_cells] = [int(i) for i in f.readline().split()]
        
        # Read node coordinates
        for k in range(n_nodes):
            node = [float(i) for i in f.readline().split()[1:]]
            nodes.append(node)

        nodes = np.array(nodes)
        
        # Read cell definitions (edges or triangles)
        for k in range(n_cells):
            cell = [int(i)-1 for i in f.readline().split()[2:]]
            if len(cell) == 2:
                edges.append(cell)
            elif len(cell) == 3:
                triangles.append(cell)

    return nodes, edges, triangles



def convert_unv_to_py(unv_file):
    """
    Reads a Universal (UNV) mesh file and converts its contents to a dictionary.
    
    Args:
        unv_file: Path to the input UNV file
        
    Returns:
        output_dict: Dictionary containing mesh data extracted from UNV file
    """
    with open(unv_file, 'r') as f:
        # File format explained at
        # http://www2.me.rochester.edu/courses/ME204/nx_help/index.html#uid:id625821
        
        # Skip header
        output_dict = {}
        for i in range(18):
            f.readline()

        # Read nodes section (2411 is UNV ID for nodes)
        line = f.readline().split()
        x = []
        y = []
        z = []

        if int(line[0]) == 2411:
            # Read node coordinates
            while line[0] != "-1":
                line = f.readline().split()
                if len(line) == 3:
                    x.append(float(line[0]))
                    y.append(float(line[1]))
                    z.append(float(line[2]))
                else:
                    continue

        output_dict["nodes"] = np.c_[x, y, z]
        line = f.readline()
        
        # Read elements section (2412 is UNV ID for elements)
        line = f.readline().split()
        all_elements = []
        edge_elements = []
        triangle_elements = []
        
        if int(line[0]) == 2412:
            while line:
                # Field 1 - element label
                # Field 2 - fe descriptor id
                # Field 3 - physical property table number
                # Field 4 - material property table number
                # Field 5 - color
                # Field 6 - number of nodes on element
                line = f.readline().split()

                if line[0] == "-1":
                    break

                fe_descriptor_id = int(line[1])
                n_nodes = int(line[-1])  # number of nodes in the element
                
                if fe_descriptor_id == 11:
                    # Beam element
                    line = f.readline()  # Skip beam orientation line
                    # Subtract 1 so first element is 0
                    cell_ids = [int(node_id)-1 for node_id in f.readline().split()]
                    if n_nodes == 2:
                        edge_elements.append(cell_ids)
                        all_elements.append(cell_ids)

                elif fe_descriptor_id == 41:
                    # Non-beam element (e.g. triangle)
                    # Subtract 1 so first element is 0
                    cell_ids = [int(node_id)-1 for node_id in f.readline().split()]
                    if n_nodes == 3:
                        triangle_elements.append(cell_ids)
                        all_elements.append(cell_ids)

        output_dict["edges"] = edge_elements
        output_dict["faces"] = triangle_elements

        line = f.readline()
        
        # Read groups section (2467 is UNV ID for groups)
        line = f.readline().split()
        if int(line[0]) == 2467:
            line = f.readline().split()
            while line:
                n_group_elements = int(line[-1])
                group_name = f.readline().split()[0]
                
                group_element_ids = []
                group_elements = []
                
                for i in range(n_group_elements):
                    ln = [float(j) for j in f.readline().split()]
                    if len(ln) == 4:
                        group_element_ids.append(ln[1]-1)
                        group_elements.append(all_elements[int(ln[1])-1])
                        break
                    elif int(ln[0]) != 8:
                        break

                    if len(ln) == 8:
                        group_element_ids.append(int(ln[1])-1)
                        group_elements.append(all_elements[int(ln[1])-1])
                        group_element_ids.append(int(ln[5])-1)
                        group_elements.append(all_elements[int(ln[5])-1])

                output_dict[group_name] = {"element_ids": group_element_ids,
                                           "elements": group_elements}

                if ln[0] == -1.0:
                    break

    return output_dict




def convert_py_to_dat(dat_filename, data):
    """
    Writes mesh data to a .dat file format.
    
    Args:
        dat_filename: Path to the output .dat file
        data: Dictionary containing nodes, edges, and triangles
    """
    nodes_dat = data["nodes"]
    edges_dat = data["edges"]
    triangles_dat = data["triangles"]
    
    with open(dat_filename, "w") as f:
        # Write header with node and cell counts
        f.write(str(len(nodes_dat))+" "+str(len(edges_dat+triangles_dat))+"\n")
        
        # Write node coordinates
        for node in enumerate(nodes_dat):
            coordinates_line = str(int(node[0]+1))
            for coord in node[1]:
                coordinates_line += " "+str(coord)
    
            coordinates_line += "\n"
            f.write(coordinates_line)
    
        # Write cell definitions
        for cells in enumerate(edges_dat+triangles_dat):
            cells_id_line = str(int(cells[0]+1))
            if len(cells[1]) == 2:
                cells_id_line +=" 102"  # Edge type
            elif len(cells[1]) == 3:
                cells_id_line +=" 203"  # Triangle type
    
            for cell_id in cells[1]:
                cells_id_line += " "+str(int(cell_id+1))
    
            cells_id_line +="\n"
            f.write(cells_id_line)




def sort_unv_for_openfoam(input_unv_file, sorted_unv_file, unv_group_name_to_map):
    """
    Sorts a specific group in a UNV file for OpenFOAM compatibility.
    
    Args:
        input_unv_file: Path to the input UNV file
        sorted_unv_file: Path to the output sorted UNV file
        unv_group_name_to_map: Name of the group to sort
    """
    f = open(input_unv_file, "r")
    mesh = f.readlines()
    f.close()

    f = open(sorted_unv_file, "w")
    new_mesh = ''

    # Search for the specified group and sort its elements
    for i in range(len(mesh)):
        if mesh[i][:-1] == unv_group_name_to_map:
            f.write(unv_group_name_to_map+"\n")
            n_plasma_faces = int(mesh[i-1].split()[-1])
            plasma_lines = []
            
            # Collect group element lines
            for k in range(i+1, i+n_plasma_faces):
                ln = [float(j) for j in mesh[k].split()]
                if len(ln) == 4:
                    plasma_lines.append(ln)
                    break
    
                elif ln[0] != 8:
                    break
    
                if len(ln) == 8:
                    plasma_lines.append(ln[:int(len(ln)/2)])
                    plasma_lines.append(ln[int(len(ln)/2):])
    
            # Sort elements by ID
            plasma_lines = np.array(plasma_lines)
            sorted_plasma_lines = plasma_lines[np.argsort(plasma_lines[:, 1])]
            n_plasma_lines = 1
            
            # Write sorted elements
            for z in range(0, len(sorted_plasma_lines), 2):
                if z == len(sorted_plasma_lines)-1:
                    f.write("         " +
                            str(int(sorted_plasma_lines[z][0]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][1]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][2]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][3]))+"\n")
                else:
                    f.write("         " +
                            str(int(sorted_plasma_lines[z][0]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][1]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][2]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z][3]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z+1][0]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z+1][1]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z+1][2]))+"         ")
                    f.write(str(int(sorted_plasma_lines[z+1][3]))+"\n")
                n_plasma_lines += 1

            # Write the rest of the file
            for new_line in range(i+n_plasma_lines, len(mesh), 1):
                f.write(mesh[new_line])
            break
        else:
            f.write(mesh[i])

    f.close()




def OF_create_hf_time_folders(t_start, t_size, hf_time_array, center_nodes, newpath, newpath_case):
    """
    Creates time step folders for heat flux (HF) data for OpenFOAM.
    
    Args:
        t_start: Starting time step
        t_size: Time step size
        hf_time_array: Array of heat flux values at different time steps
        center_nodes: Array of node coordinates
        newpath: Path for output data
        newpath_case: Path for OpenFOAM case files
    """
    _t_start = t_start
    _t_size = t_size
    _t_steps = hf_time_array.shape[1]
    times = np.arange(_t_start, _t_steps)*_t_size
    _t_sizes = _t_size
    _size_str_t_size = len(str(_t_sizes))

    # Create time folders and write heat flux data
    for i in enumerate(times):
        time_folder = i[1]
        if time_folder.is_integer():
            # Integer time folders
            if os.path.exists(os.path.join(newpath, str(int(time_folder)))):
                shutil.rmtree(os.path.join(newpath, str(int(time_folder))))
            os.makedirs(os.path.join(newpath, str(int(time_folder))))
            f = open(os.path.join(newpath, str(int(time_folder)), "HF"), "w")
            
            if os.path.exists(os.path.join(newpath_case, str(int(time_folder)))):
                shutil.rmtree(os.path.join(newpath_case, str(int(time_folder))))
            os.makedirs(os.path.join(newpath_case, str(int(time_folder))))
            f_case = open(os.path.join(newpath_case, str(int(time_folder)), "HF"), "w")
            f_case.write(HF_template1)
            f_case.write('    location    "'+str(int(time_folder))+'";\n')
            f_case.write(HF_template2)
        else:
            # Fractional time folders
            if os.path.exists(os.path.join(newpath, str(time_folder)[:_size_str_t_size])):
                shutil.rmtree(os.path.join(newpath, str(time_folder)[:_size_str_t_size]))
            os.makedirs(os.path.join(newpath, str(time_folder)[:_size_str_t_size]))
            f = open(os.path.join(newpath, str(time_folder)[:_size_str_t_size], "HF"), "w")
            
            if os.path.exists(os.path.join(newpath_case, str(time_folder)[:_size_str_t_size])):
                shutil.rmtree(os.path.join(newpath_case, str(time_folder)[:_size_str_t_size]))
            os.makedirs(os.path.join(newpath_case, str(time_folder)[:_size_str_t_size]))
            f_case = open(os.path.join(newpath_case, str(time_folder)[:_size_str_t_size], "HF"), "w")
            f_case.write(HF_template1)
            f_case.write('    location    "'+str(time_folder)[:_size_str_t_size]+'";')
            f_case.write(HF_template2)
    
        # Write heat flux values
        f.write(str(len(hf_time_array[:, i[0]]))+"\n")
        f.write("(\n")
        f_case.write(str(len(hf_time_array[:, i[0]]))+"\n")
        f_case.write("(\n")
    
        for hf in hf_time_array[:, i[0]]:
            f.write(str(hf)+"\n")
            f_case.write(str(hf)+"\n")
    
        f.write(")\n")
        f_case.write(")\n")
        f_case.write(";\n")
        f_case.write("}\n")
        f_case.write("}\n")

    # Write points file
    points_of = open(newpath+"points", 'w')
    points_of.write(str(len(hf_time_array[:, i[0]]))+"\n")
    points_of.write("(\n")
    for pt in range(len(center_nodes)):
        node = center_nodes[pt]
        points_of.write("("+str(node[0])+" "+str(node[1])+" "+str(node[2])+")\n")
    points_of.write(")\n")

    # Write temperature file for initial time
    f_T = open(os.path.join(newpath_case, str(0), "T"), "w")
    f_T.write(T_template1)


def write_Cp(new_case_folder):
    """
    Writes heat capacity data to an OpenFOAM case folder.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
    """
    f_Cp = open(os.path.join(new_case_folder, "constant", "Cp"), "w")
    f_Cp.write(Cp)
    f_Cp.close()


def write_ThermCond(new_case_folder):
    """
    Writes thermal conductivity data to an OpenFOAM case folder.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
    """
    f_thermCond = open(os.path.join(new_case_folder, "constant", "thermCond"), "w")
    f_thermCond.write(thermCond)
    f_thermCond.close()


def write_DT(new_case_folder):
    """
    Writes thermal diffusivity data to an OpenFOAM case folder.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
    """
    # Thermal diffusivity alpha = k/(Cp*rho)
    with open(os.path.join(new_case_folder, "constant", 'DT'), "w") as f_alpha:
        f_alpha.write("(")

        T = thermal_conductivity[:, 0]
        k = thermal_conductivity[:, 1]
        Cp = heat_capacity[:, 1]
        rho = density[:, 1]
        alpha = k/(Cp*rho)
        for i in range(len(alpha)):
            f_alpha.write(str(T[i])+" "+str(alpha[i])+"\n")

        f_alpha.write(")")




def write_controlDict(new_case_folder, _t_start, _t_size, hf_time_array, _t_steps_save):
    """
    Writes OpenFOAM controlDict file with simulation parameters.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
        _t_start: Starting time step
        _t_size: Time step size
        hf_time_array: Array of heat flux values at different time steps
        _t_steps_save: Frequency of writing results
    """
    _t_steps = hf_time_array.shape[1]
    times = np.arange(_t_start, _t_steps)*_t_size
    _t_sizes = _t_size

    f_controlDict = open(os.path.join(new_case_folder, "system", 'controlDict'), "w")
    f_controlDict.write(controlDict1)
    f_controlDict.write("startTime "+str(_t_start)+";\n")
    f_controlDict.write(controlDict2)
    f_controlDict.write("endTime "+str(times[-1])+";\n")
    deltaT = _t_sizes
    f_controlDict.write("deltaT "+str(deltaT)+";\n")
    f_controlDict.write(controlDict3)
    f_controlDict.write("writeInterval    "+str(_t_steps_save)+"\n")
    f_controlDict.write(controlDict4)
    f_controlDict.close()



def write_fvSchemes(new_case_folder):
    """
    Writes OpenFOAM fvSchemes file with discretization schemes.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
    """
    f_fvSchemes = open(os.path.join(new_case_folder, "system", 'fvSchemes'), "w")
    f_fvSchemes.write(fvSchemes)
    f_fvSchemes.close()


def write_fvOptions(new_case_folder):
    """
    Writes OpenFOAM fvOptions file with source terms and constraints.
    
    Args:
        new_case_folder: Path to the OpenFOAM case folder
    """
    f_fvOptions = open(os.path.join(new_case_folder, "system", 'fvOptions'), "w")
    f_fvOptions.write(fvOptions)
    f_fvOptions.close()
