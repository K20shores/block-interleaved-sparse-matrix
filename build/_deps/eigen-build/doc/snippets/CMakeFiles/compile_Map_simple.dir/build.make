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
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/flags.make

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/flags.make
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o: _deps/eigen-build/doc/snippets/compile_Map_simple.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o: _deps/eigen-src/doc/snippets/Map_simple.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o -MF CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o.d -o CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Map_simple.cpp

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Map_simple.cpp > CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.i

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Map_simple.cpp -o CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.s

# Object files for target compile_Map_simple
compile_Map_simple_OBJECTS = \
"CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o"

# External object files for target compile_Map_simple
compile_Map_simple_EXTERNAL_OBJECTS =

compile_Map_simple: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/compile_Map_simple.cpp.o
compile_Map_simple: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/build.make
compile_Map_simple: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../compile_Map_simple"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_Map_simple.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && ../../../../compile_Map_simple >/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/Map_simple.out

# Rule to build all files generated by this target.
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/build: compile_Map_simple
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/build

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_Map_simple.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/clean

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/doc/snippets /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Map_simple.dir/depend

