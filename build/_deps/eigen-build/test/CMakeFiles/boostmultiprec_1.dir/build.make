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
include _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/flags.make

_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o: _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/flags.make
_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o: _deps/eigen-src/test/boostmultiprec.cpp
_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o: _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o -MF CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o.d -o CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/boostmultiprec.cpp

_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/boostmultiprec.cpp > CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.i

_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/boostmultiprec.cpp -o CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.s

# Object files for target boostmultiprec_1
boostmultiprec_1_OBJECTS = \
"CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o"

# External object files for target boostmultiprec_1
boostmultiprec_1_EXTERNAL_OBJECTS =

boostmultiprec_1: _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/boostmultiprec.cpp.o
boostmultiprec_1: _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/build.make
boostmultiprec_1: _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../boostmultiprec_1"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/boostmultiprec_1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/build: boostmultiprec_1
.PHONY : _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/build

_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/boostmultiprec_1.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/clean

_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/test/CMakeFiles/boostmultiprec_1.dir/depend

