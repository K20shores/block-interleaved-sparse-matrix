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
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/flags.make

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/flags.make
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o: _deps/eigen-build/doc/snippets/compile_Cwise_abs2.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o: _deps/eigen-src/doc/snippets/Cwise_abs2.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o -MF CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o.d -o CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Cwise_abs2.cpp

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Cwise_abs2.cpp > CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.i

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/compile_Cwise_abs2.cpp -o CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.s

# Object files for target compile_Cwise_abs2
compile_Cwise_abs2_OBJECTS = \
"CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o"

# External object files for target compile_Cwise_abs2
compile_Cwise_abs2_EXTERNAL_OBJECTS =

compile_Cwise_abs2: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/compile_Cwise_abs2.cpp.o
compile_Cwise_abs2: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/build.make
compile_Cwise_abs2: _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../../compile_Cwise_abs2"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_Cwise_abs2.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && ../../../../compile_Cwise_abs2 >/Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/Cwise_abs2.out

# Rule to build all files generated by this target.
_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/build: compile_Cwise_abs2
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/build

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_Cwise_abs2.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/clean

_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-src/doc/snippets /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets /Users/kshores/Documents/block-interleaved-sparse-matrix/build/_deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_Cwise_abs2.dir/depend

