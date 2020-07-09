/*
 * pdal_features.hpp.in is used by cmake to generate pdal_features.hpp
 *
 * Do not edit pdal_features.hpp
 *
 */
#pragma once

/*
 * version settings
 */
#define PDAL_VERSION_MAJOR 2
#define PDAL_VERSION_MINOR 0
#define PDAL_VERSION_PATCH 1

#define PDAL_VERSION_STRING "2.0.1"

/* (note this will look yucky until we get to major>=1) */
#define PDAL_VERSION_INTEGER ((PDAL_VERSION_MAJOR*100*100)+(PDAL_VERSION_MINOR*100)+PDAL_VERSION_PATCH)

#define PDAL_PLUGIN_INSTALL_PATH "/home/chapmaca/Projects/pointcloudclassifier/envs/lib"
/*
 * availability of 3rd-party libraries
 */
#define PDAL_HAVE_LASZIP
#define PDAL_HAVE_LAZPERF
/* #undef PDAL_HAVE_HDF5 */
#define PDAL_HAVE_ZSTD
#define PDAL_HAVE_ZLIB
/* #undef PDAL_HAVE_LZMA */
#define PDAL_HAVE_LIBXML2
/* #undef PDAL_HAVE_PYTHON */

/*
 * Debug or Release build?
 */
#define PDAL_BUILD_TYPE "Release"

/*
 * built pdal app as application bundle on OSX?
 */
/* #undef PDAL_APP_BUNDLE */

