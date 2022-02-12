//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
#ifndef TORCHMLIR_CONVERSION_UTILS_H
#define TORCHMLIR_CONVERSION_UTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace torch {

Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar,
                           Type dtype);

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_UTILS_H
