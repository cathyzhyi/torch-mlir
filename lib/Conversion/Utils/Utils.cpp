//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/Utils/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"

using namespace mlir;

namespace mlir {
namespace torch {

// Convert a scalar value to the target type. The scalar value can be an element
// from a tensor or a scalar in the pytorch dialect. Both the scalar and dtype
// should be converted builtin types.
Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar,
                           Type dtype) {
  assert(isa<BuiltinDialect>(dtype.getDialect()) &&
         "scalar must be builtin type");

  Type scalarType = scalar.getType();
  assert(isa<BuiltinDialect>(scalarType.getDialect()) &&
         "scalar must be builtin type");
  if (scalarType == dtype)
    return scalar;

  // TODO: For the byte(ui8) or char(i8) case, we need the unconverted dtype to
  // be able to know if we need signed or unsigned conversion.
  auto isByteOrChar = [](Type type) {
    if (auto integerTy = type.dyn_cast<mlir::IntegerType>()) {
      return integerTy.getWidth() == 8;
    }
    return false;
  };

  if (isByteOrChar(scalarType) || isByteOrChar(dtype) ||
      dtype.isSignlessInteger(1)) {
    // TODO: Handle to-boolean conversion(from-boolean conversion is handled).
    mlir::emitError(loc)
        << "unsupported byte, char or bool type for convertScalarToDtype "
        << scalarType << "(scalar type) -> " << dtype << "(dtype)";
    return nullptr;
  }

  if (auto dtypeFloat = dtype.dyn_cast<mlir::FloatType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>()) {
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, scalar, dtype);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return b.create<arith::ExtFOp>(loc, scalar, dtype);
    }
    assert(scalarType.isa<mlir::IntegerType>());
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::UIToFPOp>(loc, scalar, dtype);
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::SIToFPOp>(loc, scalar, dtype);
  }

  if (auto dtypeInteger = dtype.dyn_cast<mlir::IntegerType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>())
      return b.create<arith::FPToSIOp>(loc, scalar, dtype);
    assert(scalarType.isa<mlir::IntegerType>());
    auto scalarInteger = scalarType.cast<mlir::IntegerType>();
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return b.create<arith::TruncIOp>(loc, scalar, dtype);
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::ExtUIOp>(loc, scalar, dtype);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::ExtSIOp>(loc, scalar, dtype);
  }

  llvm_unreachable("convertScalarToDtype should handle all the types");
}

} // namespace torch
} // namespace mlir
