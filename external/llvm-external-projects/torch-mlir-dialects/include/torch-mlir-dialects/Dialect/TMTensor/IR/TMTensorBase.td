//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_BASE
#define TORCH_MLIR_DIALECT_TMTENSOR_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def TMTensor_Dialect : Dialect {
  let name = "tm_tensor";
  let cppNamespace = "::mlir::torch::TMTensor";
  let description = [{
    The `tm_tensor` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // TORCH_MLIR_DIALECT_TMTENSOR_BASE
