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
include _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/flags.make

_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o: _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/flags.make
_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o: _deps/eigen-src/test/packetmath.cpp
_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o: _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o -MF CMakeFiles/packetmath_5.dir/packetmath.cpp.o.d -o CMakeFiles/packetmath_5.dir/packetmath.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/packetmath.cpp

_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/packetmath_5.dir/packetmath.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/packetmath.cpp > CMakeFiles/packetmath_5.dir/packetmath.cpp.i

_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/packetmath_5.dir/packetmath.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test/packetmath.cpp -o CMakeFiles/packetmath_5.dir/packetmath.cpp.s

# Object files for target packetmath_5
packetmath_5_OBJECTS = \
"CMakeFiles/packetmath_5.dir/packetmath.cpp.o"

# External object files for target packetmath_5
packetmath_5_EXTERNAL_OBJECTS =

packetmath_5: _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/packetmath.cpp.o
packetmath_5: _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/build.make
packetmath_5: _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../packetmath_5"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/packetmath_5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/build: packetmath_5
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/build

_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/packetmath_5.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/clean

_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/test/CMakeFiles/packetmath_5.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_5.dir/depend

