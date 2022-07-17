cmake_minimum_required(VERSION 3.4)

# default to a release build (must be before project())
if (DEFINED CMAKE_BUILD_TYPE)
  set(DEF_BUILD_TYPE ${CMAKE_BUILD_TYPE})
else()
  set(DEF_BUILD_TYPE Release)
endif()
set(CMAKE_BUILD_TYPE ${DEF_BUILD_TYPE} CACHE STRING "Build type: <none>|Debug|Release|RelWithDebInfo|MinSizeRel")

# initialize our project
project(cluttercal)

# setup standard install locations
include(GNUInstallDirs)

# external dependencies
find_package(bom-util REQUIRED)
list(APPEND DEPENDENCY_LIBRARIES bom-util)

find_package(bom-core REQUIRED)
list(APPEND DEPENDENCY_LIBRARIES bom-core)

# setup our compilation environment
setup_cplusplus()

# build our executables
add_executable(layered-flow src/main.cc)
target_link_libraries(layered-flow ${DEPENDENCY_LIBRARIES} stdc++fs)
install(TARGETS layered-flow DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime)
