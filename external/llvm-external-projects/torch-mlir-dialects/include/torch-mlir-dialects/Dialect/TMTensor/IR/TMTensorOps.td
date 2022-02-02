//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_OPS
#define TORCH_MLIR_DIALECT_TMTENSOR_OPS

include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorBase.td"
include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorInterfaces.td"
include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class TMTensor_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<TMTensor_Dialect, mnemonic, traits> {
}

class TMTensor_Op<string mnemonic, list<OpTrait> traits = []> :
    TMTensor_PureOp<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         TMTensorInterface,
         SingleBlockImplicitTerminator<"::mlir::torch::TMTensor::YieldOp">
  ])> {
  let verifier = [{ return verify$cppClass(*this); }];
  let printer = [{ return print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  code extraTMTensorOpClassDeclaration = [{
    SmallVector<Value> getDestinationOperands(OpBuilder &b) {
      SmallVector<Value> dest(outputs().begin(), outputs().end());
      return dest;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Non-structured ops
//===----------------------------------------------------------------------===//

def TMTensor_ScanOp : TMTensor_Op<"scan"
         ,[DeclareOpInterfaceMethods<ScalarLoopOpInterface,
           ["generateScalarImplementation"]>]> {
  let summary = "Scan operator";
  let description = [{
    Computes the inclusive/exclusive scan along a given dimension.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64Attr:$dimension,
                       BoolAttr:$inclusive
  );

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"int64_t", "0">:$dimension, CArg<"bool", "true">:$inclusive)>
  ];

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region AnyRegion:$region);
  let hasFolder = 1;
  let assemblyFormat = [{
    `dimension` `(` $dimension `)`
    `inclusive` `(` $inclusive `)`
    attr-dict
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = extraTMTensorOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value accumulator() {
      return getOutputOperand(1)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Pure ops
//===----------------------------------------------------------------------===//

def TMTensor_YieldOp : TMTensor_PureOp<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "TMTensor yield op";
  let description = [{
    `tm_tensor.yield` is a special terminator operation for blocks inside
    regions in `tm_tensor` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif  // TORCH_MLIR_DIALECT_TMTENSOR_OPS
