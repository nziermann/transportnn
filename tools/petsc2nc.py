#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import netCDF4 as nc4
import math
import yaml

#
#   read_PETSc_vec
#
def read_PETSc_vec(file):
    # debug
#    print("read_PETSc_vec ... %s" % file)
    # open file
    # omit header
    # read length
    # read values
    # close file
    f = open(file, "rb")
    np.fromfile(f, dtype=">i4", count=1)
    nvec, = np.fromfile(f, dtype=">i4", count=1)
    v = np.fromfile(f, dtype=">f8", count=nvec)
    f.close()
    return v

#
#   read_conf_file
#
def read_conf_file(conf_file):
    print("Reading configuration file ... " + conf_file)
    # open conf file
    f = open(conf_file, "r")
    # parse yaml file
    conf = yaml.load(f)
    # get list of variables
    conf_list = []
    try:
        var_list = conf["Name, Scale, Unit, Description"]
        # loop over list
        for var in var_list:
            # split
            name, scale, unit, description = var.split(",", 3)
            # strip and convert
            name = name.strip()
            scale = float(scale.strip())
            unit = unit.strip()
            description = description.strip()
            # append to conf list
            conf_list.append({"name": name, "scale": scale, "unit": unit, "description": description})
    except KeyError:
        print("### ERROR ### Did not find the 'Name, Scale, Unit, Description' key.")
        sys.exit(1)
    # return results
    return conf_list

#
#   write_netcdf_file
#
def write_netcdf_file(vardim, num_time_steps, grid_file, conf_list, petsc_data, out_netcdf_file):
    print("Writing NetCDF file ... " + out_netcdf_file)

    # open grid file
    grid_nc4 = nc4.Dataset(grid_file, "r")
    # netcdf variable
    try:
        grid_mask_variable = grid_nc4.variables["grid_mask"]
    except KeyError:
        print("### ERROR ### No 'grid_mask' variable found.")
        sys.exit(1)

    # get sizes, we expect  (..., y, x)
    if vardim == "2d":
        # numpy masked array
        grid_mask = grid_mask_variable[0,0,:,:]
        ny, nx = grid_mask.shape
        print("Grid mask 2D ... ", "ny:", ny, "nx:", nx)
    elif vardim == "3d":
        # numpy masked array
        grid_mask = grid_mask_variable[0,:,:,:]
        nz, ny, nx = grid_mask.shape
        print("Grid mask 3D ... ", "nz:", nz, "ny:", ny, "nx:", nx)
    else:
        print("### ERROR ### Unknown dimension: " + dim)
        sys.exit(1)

    # create netcdf file
    out_file = nc4.Dataset(out_netcdf_file, "w", format = "NETCDF4")
    # set usage of fill value
    out_file.set_fill_on()
    # create global attributes
    out_file.description = "Metos3D tracer file for 2.8125 degree, 15 layers MITgcm resolution"
    out_file.history = "created with:" + " %s"*len(sys.argv) % tuple(sys.argv)

    # copy dimensions from grid file
    for dimname, dim in grid_nc4.dimensions.items():
        #print(dimname)
        dim_length = len(dim)
        if(dimname == 'time'):
            dim_length = num_time_steps
        out_file.createDimension(dimname, dim_length)
        out_file.sync()

    # copy variables)
    for varname, ncvar in grid_nc4.variables.items():
#        print(varname)
        if varname in ["time", "depth", "lat", "lon"]:
            var = out_file.createVariable(varname, ncvar.dtype, ncvar.dimensions, zlib = True, fill_value = -9.e+33)
            attdict = ncvar.__dict__
            var.setncatts(attdict)
            if varname == 'time':
                ncvar = np.zeros(num_time_steps)
                ncvar[:] = range(0, num_time_steps)
            var[:] = ncvar[:]
            out_file.sync()

    print(vardim)
    # create variables
    if vardim == "2d":
        # 2d
        work_array = grid_mask.flatten()
        for var_list in conf_list:
            print(var_list)
            var = out_file.createVariable(var_list["name"], "f8", ("time", "lat", "lon", ), zlib = True, fill_value = -9.e+33)
            var.unit = var_list["unit"]
            var.description = var_list["description"]
            # transform from 1d to 2d
            work_array[~work_array.mask] = petsc_data[var_list["name"]] * var_list["scale"]
            var[:,:,:] = work_array.reshape(num_time_steps, ny, nx)

    elif vardim == "3d":
        # 3d
        work_array = grid_mask.reshape((nz, ny*nx)).transpose().flatten()
        #data = np.zeros((np.size(work_array), num_time_steps))
        data = np.zeros((num_time_steps, np.size(work_array)))
        for var_list in conf_list:
            print("Var list: ", var_list)
            var = out_file.createVariable(var_list["name"], "f8", ("time", "depth", "lat", "lon", ), zlib = True, fill_value = -9.e+33)
            var.unit = var_list["unit"]
            var.description = var_list["description"]
            # transform from 1d to 3d
            print(np.shape(data))
            print(np.shape(work_array))
            print(np.shape(petsc_data[var_list["name"]]))
            #data[~work_array.mask,:] = petsc_data[var_list["name"]] * var_list["scale"]
            data[:, ~work_array.mask] = petsc_data[var_list["name"]] * var_list["scale"]
            #var[:,:,:,:] = data.reshape(ny*nx, nz, num_time_steps).transpose().reshape(nz, ny, nx, num_time_steps)
            #var[:,:,:,:] = data.reshape((num_time_steps, ny*nx, nz)).transpose().reshape((num_time_steps, nz, ny, nx))
            var[:,:,:,:] = data.reshape((num_time_steps, ny*nx, nz)).transpose((0, 2, 1)).reshape((num_time_steps, nz, ny, nx))


    print(var)
    # close file
    out_file.close()

#
#   get_data_from_petsc_file
#
def get_data_from_petsc_file(conf_list, num_time_steps):
    print("Opening PETSc files ...")
    # loop over names
    petsc_data = {}
    padding_length = math.ceil(math.log(num_time_steps, 10))
    for conf in conf_list:
        first_file = "work/{0:0>{padding_length}}-{1}.petsc".format(0, conf["name"].upper(), padding_length=padding_length)
        f = open(first_file, "rb")
        np.fromfile(f, dtype=">i4", count=1)
        nvec, = np.fromfile(f, dtype=">i4", count=1)
        f.close()

        #petsc_data[conf["name"]] = np.zeros((nvec, num_time_steps))
        petsc_data[conf["name"]] = np.zeros((num_time_steps, nvec))
        for i in range(0, num_time_steps):
            petsc_file = "work/{0:0>{padding_length}}-{1}.petsc".format(i, conf["name"].upper(), padding_length=padding_length)
            # read petsc file
            petsc_vec = read_PETSc_vec(petsc_file)
            #store
            #petsc_data[conf["name"]][:,i] = petsc_vec
            petsc_data[conf["name"]][i, :] = petsc_vec
            #print("Read data from: ", petsc_file)
        print("Read data for: ", conf["name"])

    return petsc_data

#
#   main
#
if __name__ == "__main__":
    # no arguments?
    if len(sys.argv) <= 5:
        # print usage and exit with code 1
        print("usage: %s [2d|3d] [num_time_steps] [grid-netcdf-file] [conf-yaml-file] [out-netcdf-file]" % sys.argv[0])
        sys.exit(1)
    # dim
    vardim = sys.argv[1]
    #dim time
    num_time_steps = int(sys.argv[2])
    # grid file
    grid_file = sys.argv[3]
    # conf yaml file
    conf_yaml_file = sys.argv[4]
    # out netcdf file
    out_netcdf_file = sys.argv[5]
    # debug
#    print(conf_yaml_file, out_netcdf_file)
    # read conf file
    conf_list = read_conf_file(conf_yaml_file)
    # debug
#    print(conf_list)
    # get data from petsc file
    petsc_data = get_data_from_petsc_file(conf_list, num_time_steps)
    # debug
#    print(petsc_data)
    # write netcdf file
    write_netcdf_file(vardim, num_time_steps, grid_file, conf_list, petsc_data, out_netcdf_file)