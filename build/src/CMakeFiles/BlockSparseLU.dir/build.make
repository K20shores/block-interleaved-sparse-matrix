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
include src/CMakeFiles/BlockSparseLU.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/BlockSparseLU.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/BlockSparseLU.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/BlockSparseLU.dir/flags.make

src/CMakeFiles/BlockSparseLU.dir/main.cpp.o: src/CMakeFiles/BlockSparseLU.dir/flags.make
src/CMakeFiles/BlockSparseLU.dir/main.cpp.o: /Users/kshores/Documents/block-interleaved-sparse-matrix/src/main.cpp
src/CMakeFiles/BlockSparseLU.dir/main.cpp.o: src/CMakeFiles/BlockSparseLU.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/BlockSparseLU.dir/main.cpp.o"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/BlockSparseLU.dir/main.cpp.o -MF CMakeFiles/BlockSparseLU.dir/main.cpp.o.d -o CMakeFiles/BlockSparseLU.dir/main.cpp.o -c /Users/kshores/Documents/block-interleaved-sparse-matrix/src/main.cpp

src/CMakeFiles/BlockSparseLU.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/BlockSparseLU.dir/main.cpp.i"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/kshores/Documents/block-interleaved-sparse-matrix/src/main.cpp > CMakeFiles/BlockSparseLU.dir/main.cpp.i

src/CMakeFiles/BlockSparseLU.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/BlockSparseLU.dir/main.cpp.s"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/kshores/Documents/block-interleaved-sparse-matrix/src/main.cpp -o CMakeFiles/BlockSparseLU.dir/main.cpp.s

# Object files for target BlockSparseLU
BlockSparseLU_OBJECTS = \
"CMakeFiles/BlockSparseLU.dir/main.cpp.o"

# External object files for target BlockSparseLU
BlockSparseLU_EXTERNAL_OBJECTS =

BlockSparseLU: src/CMakeFiles/BlockSparseLU.dir/main.cpp.o
BlockSparseLU: src/CMakeFiles/BlockSparseLU.dir/build.make
BlockSparseLU: src/libbism.a
BlockSparseLU: src/CMakeFiles/BlockSparseLU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/kshores/Documents/block-interleaved-sparse-matrix/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../BlockSparseLU"
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BlockSparseLU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/BlockSparseLU.dir/build: BlockSparseLU
.PHONY : src/CMakeFiles/BlockSparseLU.dir/build

src/CMakeFiles/BlockSparseLU.dir/clean:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src && $(CMAKE_COMMAND) -P CMakeFiles/BlockSparseLU.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/BlockSparseLU.dir/clean

src/CMakeFiles/BlockSparseLU.dir/depend:
	cd /Users/kshores/Documents/block-interleaved-sparse-matrix/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kshores/Documents/block-interleaved-sparse-matrix /Users/kshores/Documents/block-interleaved-sparse-matrix/src /Users/kshores/Documents/block-interleaved-sparse-matrix/build /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src /Users/kshores/Documents/block-interleaved-sparse-matrix/build/src/CMakeFiles/BlockSparseLU.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/BlockSparseLU.dir/depend

