# Try to find VAST headers and libraries.
#
# Use this module as follows:
#
#     find_package(VAST)
#
# Variables used by this module (they can change the default behaviour and need
# to be set before calling find_package):
#
#  VAST_ROOT_DIR  Set this variable either to an installation prefix or to wa
#                VAST build directory where to look for the VAST libraries.
#
# Variables defined by this module:
#
#  VAST_FOUND              System has VAST headers and library
#  VAST_LIBRARIES          List of library files  for all components
#  VAST_INCLUDE_DIRS       List of include paths for all components

# iterate over user-defined components

set(header_hints
    "${VAST_ROOT_DIR}/include"
    "${VAST_ROOT_DIR}/../libvast"
    "${VAST_ROOT_DIR}/libvast")
find_path(VAST_INCLUDE_DIR
          NAMES
            "vast/bitmap_index.hpp"
          HINTS
            ${header_hints}
            /usr/include
            /usr/local/include
            /opt/local/include
            /sw/include
            ${CMAKE_INSTALL_PREFIX}/include)
mark_as_advanced(VAST_INCLUDE_DIR)
set(VAST_INCLUDE_DIRS ${VAST_INCLUDE_DIRS} ${VAST_INCLUDE_DIR})
if (NOT "${VAST_INCLUDE_DIR}" STREQUAL "VAST_INCLUDE_DIR-NOTFOUND")
  find_path(VAST_INCLUDE_CONF_DIR
        NAMES
          "vast/config.hpp"
        HINTS
          ${header_hints}
          /usr/include
          /usr/local/include
          /opt/local/include
          /sw/include
          ${CMAKE_INSTALL_PREFIX}/include)
  if (NOT "${VAST_INCLUDE_DIR}" STREQUAL "VAST_INCLUDE_DIR-NOTFOUND")
    # mark as found (set back to false in case library cannot be found)
    set(VAST_FOUND true)
    # add to VAST_INCLUDE_CONF_DIRS only if path isn't already set
    set(duplicate false)
    foreach (p ${VAST_INCLUDE_DIRS})
      if (${p} STREQUAL ${VAST_INCLUDE_CONF_DIR})
        set(duplicate true)
      endif ()
    endforeach ()
    if (NOT duplicate)
      set(VAST_INCLUDE_DIRS ${VAST_INCLUDE_DIRS} ${VAST_INCLUDE_CONF_DIR})
    endif()
    mark_as_advanced(VAST_INCLUDE_CONF_DIR)
    # look for (.dll|.so|.dylib) file, again giving hints for non-installed VASTs
    # skip probe_event as it is header only
    if (VAST_ROOT_DIR)
      set(library_hints "${VAST_ROOT_DIR}/lib")
    endif ()
    find_library(VAST_LIBRARY
                 NAMES
                   "vast"
                 HINTS
                   ${library_hints}
                   /usr/lib
                   /usr/local/lib
                   /opt/local/lib
                   /sw/lib
                   ${CMAKE_INSTALL_PREFIX}/lib)
    mark_as_advanced(VAST_LIBRARY)
    if ("${VAST_LIBRARY}" STREQUAL "VAST_LIBRARY-NOTFOUND")
      set(VAST_FOUND false)
    else ()
      set(VAST_LIBRARIES ${VAST_LIBRARIES} ${VAST_LIBRARY})
    endif ()
  endif ()
endif ()

# let CMake check whether all requested components have been found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VAST
                                  FOUND_VAR VAST_FOUND
                                  REQUIRED_VARS VAST_LIBRARIES VAST_INCLUDE_DIRS
                                  HANDLE_COMPONENTS)

# final step to tell CMake we're done
mark_as_advanced(VAST_ROOT_DIR
                 VAST_LIBRARIES
                 VAST_INCLUDE_DIRS)

