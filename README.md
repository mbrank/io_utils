# IO utils
Python IO routines for VTK, DAT file formats


# Mesh Processing Utilities

This library provides a comprehensive set of functions for processing mesh data, performing geometric calculations, file format conversions, and preparing thermal simulation data for OpenFOAM.

## Table of Contents

- [Geometry Processing Functions](#geometry-processing-functions)
- [Mesh Conversion Functions](#mesh-conversion-functions)
- [Mesh Data Manipulation Functions](#mesh-data-manipulation-functions)
- [VTK File Generation Functions](#vtk-file-generation-functions)
- [File Format Conversion Functions](#file-format-conversion-functions)
- [OpenFOAM Integration Functions](#openfoam-integration-functions)

## Function Descriptions

### Geometry Processing Functions

| Function | Description |
|----------|-------------|
| `check_clockwise_order(nodes, cells)` | Determines if cells (quadrilaterals) have clockwise or counter-clockwise orientation. Returns an array with values -1 for counter-clockwise or 1 for clockwise orientation. |
| `calc_node_distance(pt0, pt1)` | Calculates the Euclidean distance between two points in 2D. |
| `calc_max_quad_area(a, b, c, d)` | Calculates the maximum possible area of a quadrilateral using Brahmagupta's formula based on side lengths. |
| `calc_barycenter(x1, y1, x2, y2, x3, y3)` | Calculates the barycenter (centroid) of a triangle in 2D. |
| `calc_barycenter3D(x1, y1, z1, x2, y2, z2, x3, y3, z3)` | Calculates the barycenter (centroid) of a triangle in 3D. |
| `calc_cells_barycenter3D(nodes, cells)` | Calculates the barycenters of triangular cells in a 3D mesh. |
| `calc_tria_area(x1, y1, x2, y2, x3, y3)` | Calculates the area of a triangle in 2D using the shoelace formula. |
| `find_centroid_of_4_coordinates(a, b, c, d)` | Finds the centroid of a quadrilateral by splitting it into two triangles and computing their centroids. |
| `calc_cell_centroid(nodes, cells)` | Calculates centroids for each cell in a mesh. Returns arrays of x and y coordinates of cell centroids. |
| `quadrangle_centroid_shoelace_area(pt_x, pt_y)` | Calculates the centroid of a quadrilateral using the shoelace formula. |
| `calc_cells_area(nodes, cells)` | Calculates the area of each cell in a mesh using the shoelace formula. |
| `calc_shoelace_area(x_list, y_list)` | Calculates the area of a polygon using the shoelace formula. |

### Mesh Conversion Functions

| Function | Description |
|----------|-------------|
| `convert_quad2tria(cells)` | Converts quadrilateral cells to triangular cells by splitting each quad into two triangles. |
| `convert_quaddata2triadata(_data)` | Converts data associated with quadrilateral cells to corresponding triangular cell data. |
| `convert_node_data_to_cell_data(nodes, cells, node_data)` | Converts data associated with nodes to cell data by averaging node values for each cell. |

### Mesh Data Manipulation Functions

| Function | Description |
|----------|-------------|
| `remove_orphan_nodes(data)` | Removes nodes that are not referenced by any cell from the mesh data. |
| `merge_two_triangular_meshes(vtk_data0, vtk_data1)` | Merges two triangular meshes into a single mesh. |
| `remove_double_nodes(data)` | Identifies and removes duplicate nodes from mesh data. |

### VTK File Generation Functions

| Function | Description |
|----------|-------------|
| `generate_tetra_vtk(file_name, vtk_data)` | Generates a VTK file for tetrahedral mesh elements. |
| `generate_faces_vtk(file_name, vtk_data)` | Generates a VTK file for triangular or quadrilateral face elements. |
| `generate_edges_vtk(file_name, vtk_data)` | Generates a VTK file for edge elements. |
| `generate_points_vtk(file_name, vtk_data)` | Generates a VTK file for point data only without any connectivity information. |
| `read_vtk(file_name)` | Reads a VTK file and converts its contents to a dictionary. |

### File Format Conversion Functions

| Function | Description |
|----------|-------------|
| `convert_dat_to_py(dat_file)` | Reads a .dat mesh file and extracts nodes, edges, and triangles. |
| `convert_unv_to_py(unv_file)` | Reads a Universal (UNV) mesh file and converts its contents to a dictionary. |
| `convert_py_to_dat(dat_filename, data)` | Writes mesh data to a .dat file format. |
| `sort_unv_for_openfoam(input_unv_file, sorted_unv_file, unv_group_name_to_map)` | Sorts a specific group in a UNV file for OpenFOAM compatibility. |

### OpenFOAM Integration Functions

| Function | Description |
|----------|-------------|
| `OF_create_hf_time_folders(t_start, t_size, hf_time_array, center_nodes, newpath, newpath_case)` | Creates time step folders for heat flux (HF) data for OpenFOAM. |
| `write_Cp(new_case_folder)` | Writes heat capacity data to an OpenFOAM case folder. |
| `write_ThermCond(new_case_folder)` | Writes thermal conductivity data to an OpenFOAM case folder. |
| `write_DT(new_case_folder)` | Writes thermal diffusivity data to an OpenFOAM case folder based on the formula alpha = k/(Cp*rho). |
| `write_controlDict(new_case_folder, _t_start, _t_size, hf_time_array, _t_steps_save)` | Writes OpenFOAM controlDict file with simulation parameters. |
| `write_fvSchemes(new_case_folder)` | Writes OpenFOAM fvSchemes file with discretization schemes. |
| `write_fvOptions(new_case_folder)` | Writes OpenFOAM fvOptions file with source terms and constraints. |
| `write_fvSolution(new_case_folder)` | Writes OpenFOAM fvSolution file with solver settings and solution control parameters. |

## Usage

```python
import mesh_processing as mp

# Example: Convert a quad mesh to triangular mesh
triangular_cells = mp.convert_quad2tria(quadrilateral_cells)

# Example: Generate a VTK file for visualization
mp.generate_faces_vtk("output.vtk", mesh_data)

# Example: Prepare OpenFOAM case for thermal simulation
mp.write_controlDict(case_dir, 0, 0.1, heat_flux_data, 10)
```

## Dependencies

- numpy
- math
- copy
- os
- shutil

## Note

The OpenFOAM integration functions assume the presence of template variables (HF_template1, controlDict1, etc.) which should be defined elsewhere in your project.
