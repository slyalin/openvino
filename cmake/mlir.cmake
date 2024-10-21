# Allow MLIR_DIR to be set via command line
if(DEFINED MLIR_DIR)
  message(STATUS "************************* Using pre-built MLIR from: ${MLIR_DIR}")
  #list(APPEND CMAKE_PREFIX_PATH ${MLIR_DIR})

  # Find MLIR package
  find_package(MLIR REQUIRED CONFIG)

else()
  # FetchContent block to download and build MLIR if no pre-built MLIR is provided
  message(STATUS "********************** Fetching and building MLIR...")

  include(FetchContent)

  FetchContent_Declare(
    mlir
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG        bf684034844c660b778f0eba103582f582b710c9
    SOURCE_SUBDIR  mlir
  )

  # Make the MLIR project available
  FetchContent_MakeAvailable(mlir)

endif()
