# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kshores/Documents/block-interleaved-sparse-matrix

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kshores/Documents/block-interleaved-sparse-matrix/build

# Include any dependencies generated for this target.
include _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/flags.make

_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.o: _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/flags.make
_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.o: _deps/eigen-src/blas/testing/sblat1.f
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing && /opt/homebrew/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/blas/testing/sblat1.f -o CMakeFiles/sblat1.dir/sblat1.f.o

_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing Fortran source to CMakeFiles/sblat1.dir/sblat1.f.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing && /opt/homebrew/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/blas/testing/sblat1.f > CMakeFiles/sblat1.dir/sblat1.f.i

_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling Fortran source to assembly CMakeFiles/sblat1.dir/sblat1.f.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing && /opt/homebrew/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/blas/testing/sblat1.f -o CMakeFiles/sblat1.dir/sblat1.f.s

# Object files for target sblat1
sblat1_OBJECTS = \
"CMakeFiles/sblat1.dir/sblat1.f.o"

# External object files for target sblat1
sblat1_EXTERNAL_OBJECTS =

sblat1: _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/sblat1.f.o
sblat1: _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/build.make
sblat1: _deps/eigen-build/blas/libeigen_blas.dylib
sblat1: _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking Fortran executable ../../../../sblat1"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sblat1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/build: sblat1
.PHONY : _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/build

_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing && $(CMAKE_COMMAND) -P CMakeFiles/sblat1.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/clean

_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/blas/testing /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/blas/testing/CMakeFiles/sblat1.dir/depend

