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
include _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/flags.make

_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o: _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/flags.make
_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o: _deps/eigen-src/unsupported/doc/examples/PolynomialUtils1.cpp
_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o: _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o -MF CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o.d -o CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/doc/examples/PolynomialUtils1.cpp

_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/doc/examples/PolynomialUtils1.cpp > CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.i

_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/doc/examples/PolynomialUtils1.cpp -o CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.s

# Object files for target example_PolynomialUtils1
example_PolynomialUtils1_OBJECTS = \
"CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o"

# External object files for target example_PolynomialUtils1
example_PolynomialUtils1_EXTERNAL_OBJECTS =

example_PolynomialUtils1: _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/PolynomialUtils1.cpp.o
example_PolynomialUtils1: _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/build.make
example_PolynomialUtils1: _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../../example_PolynomialUtils1"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_PolynomialUtils1.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && ../../../../../example_PolynomialUtils1 >/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples/PolynomialUtils1.out

# Rule to build all files generated by this target.
_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/build: example_PolynomialUtils1
.PHONY : _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/build

_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/example_PolynomialUtils1.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/clean

_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/unsupported/doc/examples /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/unsupported/doc/examples/CMakeFiles/example_PolynomialUtils1.dir/depend

