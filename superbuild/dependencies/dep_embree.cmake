## Copyright 2009 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(COMPONENT_NAME embree)

set(COMPONENT_PATH ${CMAKE_INSTALL_PREFIX})

if (EMBREE_HASH)
  set(EMBREE_URL_HASH URL_HASH SHA256=${EMBREE_HASH})
endif()

if (BUILD_EMBREE_FROM_SOURCE)
  if(${EMBREE_VERSION} MATCHES "(^[0-9]+\.[0-9]+\.[0-9]+$)")
    set(EMBREE_DEFAULT_URL "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VERSION}/embree-${EMBREE_VERSION}.src.tar.gz")
  else()
    set(EMBREE_DEFAULT_URL "https://www.github.com/RenderKit/embree.git")
  endif()
    set(EMBREE_URL ${OIDN_DEFAULT_URL} CACHE STRING "Location to get Embree source from")
  if (${EMBREE_URL} MATCHES ".*\.src\.tar\.gz$")
    set(EMBREE_CLONE_URL URL ${EMBREE_URL})
  else()
    set(EMBREE_CLONE_URL GIT_REPOSITORY ${EMBREE_URL} GIT_TAG ${EMBREE_VERSION})
  endif()

  #if(APPLE)
  #  set(OIDN_PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/../patches/oidn-2.3.0-patch.patch)
  #endif()

  ExternalProject_Add(${COMPONENT_NAME}
    PREFIX ${COMPONENT_NAME}
    DOWNLOAD_DIR ${COMPONENT_NAME}
    STAMP_DIR ${COMPONENT_NAME}/stamp
    SOURCE_DIR ${COMPONENT_NAME}/src
    BINARY_DIR ${COMPONENT_NAME}/build
    LIST_SEPARATOR | # Use the alternate list separator
    ${OIDN_CLONE_URL}
    ${OIDN_URL_HASH}
    GIT_SHALLOW ON
    CMAKE_ARGS
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_INSTALL_PREFIX:PATH=${COMPONENT_PATH}
      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
      -DCMAKE_INSTALL_DOCDIR=${CMAKE_INSTALL_DOCDIR}
      -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
      $<$<BOOL:${BUILD_TBB}>:-DTBB_ROOT=${COMPONENT_PATH}>
      $<$<BOOL:${DOWNLOAD_ISPC}>:-DISPC_EXECUTABLE=${ISPC_PATH}>
      -DCMAKE_BUILD_TYPE=Release # XXX debug builds are currently broken
      -DOIDN_APPS=OFF
      -DOIDN_ZIP_MODE=ON # to set install RPATH
      -DOIDN_DEVICE_SYCL=${BUILD_GPU_SUPPORT}
      -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
      -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    PATCH_COMMAND ${OIDN_PATCH_COMMAND}
    BUILD_COMMAND ${DEFAULT_BUILD_COMMAND}
    BUILD_ALWAYS ${ALWAYS_REBUILD}
  )

  if (BUILD_TBB)
    ExternalProject_Add_StepDependencies(${COMPONENT_NAME} configure dep_tbb)
  endif()

  if (DOWNLOAD_TBB)
    ExternalProject_Add_StepDependencies(${COMPONENT_NAME} configure tbb)
  endif()

else()
#https://github.com/RenderKit/embree/releases/download/v4.4.0/embree-4.4.0.arm64.macosx.zip
  if (APPLE)
    set(EMBREE_OSSUFFIX "${CMAKE_SYSTEM_PROCESSOR}.macosx.zip")
  elseif (WIN32)
    set(EMBREE_OSSUFFIX "x64.windows.zip")
  else()
    set(EMBREE_OSSUFFIX "x86_64.linux.tar.gz")
  endif()
  set(EMBREE_URL "https://github.com/RenderKit/embree/releases/download/v${EMBREE_VERSION}/embree-${EMBREE_VERSION}.${EMBREE_OSSUFFIX}")

message(STATUS "EMBREE_URL = ${EMBREE_URL}")

  ExternalProject_Add(${COMPONENT_NAME}
    PREFIX ${COMPONENT_NAME}
    DOWNLOAD_DIR ${COMPONENT_NAME}
    STAMP_DIR ${COMPONENT_NAME}/stamp
    SOURCE_DIR ${COMPONENT_NAME}/src
    BINARY_DIR ${COMPONENT_NAME}
    URL ${EMBREE_URL}
    ${EMBREE_URL_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "${CMAKE_COMMAND}" -E copy_directory
      <SOURCE_DIR>/
      ${COMPONENT_PATH}
    BUILD_ALWAYS OFF
  )
endif()

list(APPEND CMAKE_PREFIX_PATH ${COMPONENT_PATH})
string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")