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
include _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/flags.make

_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o: _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/flags.make
_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o: _deps/eigen-src/test/mapped_matrix.cpp
_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o: _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o -MF CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o.d -o CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/mapped_matrix.cpp

_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/mapped_matrix.cpp > CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.i

_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/mapped_matrix.cpp -o CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.s

# Object files for target mapped_matrix_10
mapped_matrix_10_OBJECTS = \
"CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o"

# External object files for target mapped_matrix_10
mapped_matrix_10_EXTERNAL_OBJECTS =

mapped_matrix_10: _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/mapped_matrix.cpp.o
mapped_matrix_10: _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/build.make
mapped_matrix_10: _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../mapped_matrix_10"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mapped_matrix_10.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/build: mapped_matrix_10
.PHONY : _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/build

_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/mapped_matrix_10.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/clean

_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/test/CMakeFiles/mapped_matrix_10.dir/depend

