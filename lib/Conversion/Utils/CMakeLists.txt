add_mlir_dialect_library(TorchMLIRConversionUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/Utils
  )
