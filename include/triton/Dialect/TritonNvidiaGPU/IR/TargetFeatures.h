#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace mlir::triton::nvidia_gpu {

class TargetFeatures {
public:
  explicit TargetFeatures(int computeCapability)
      : computeCapability(computeCapability), acceleratedFeatures(false) {}
  TargetFeatures(int computeCapability, bool acceleratedFeatures)
      : computeCapability(computeCapability),
        acceleratedFeatures(acceleratedFeatures) {}

  static TargetFeatures fromModuleOp(ModuleOp moduleOp) {
    auto targetAttr =
        moduleOp->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
    assert(targetAttr && "Expected a target attribute on the module operation");

    StringRef targetName = targetAttr.getValue();
    assert(targetName.starts_with(kTargetPrefix) &&
           "expected target attribute to be prefixed with \"cuda:\"");

    StringRef arch = targetName.drop_front(sizeof(kTargetPrefix) - 1);
    bool acceleratedFeatures = false;
    if (arch.starts_with("sm_"))
      arch = arch.drop_front(3);
    else if (arch.starts_with("sm"))
      arch = arch.drop_front(2);
    else
      llvm::report_fatal_error(
          "expected CUDA target attribute to use sm<capability>[a]");

    if (arch.ends_with("a")) {
      acceleratedFeatures = true;
      arch = arch.drop_back(1);
    }

    int computeCapability;
    bool parseError = arch.getAsInteger(10, computeCapability);
    if (parseError)
      llvm::report_fatal_error(
          "invalid compute capability string in target attribute");

    return TargetFeatures(computeCapability, acceleratedFeatures);
  }

  int getComputeCapability() const { return computeCapability; }
  bool hasAcceleratedFeatures() const { return acceleratedFeatures; }

  bool supportClusterOps() const {
    return computeCapability >= 90 && computeCapability / 10 != 12;
  }

  bool supportMMA3() const {
    return computeCapability >= 90 && computeCapability < 120 &&
           acceleratedFeatures;
  }

  bool supportMMA5() const {
    return computeCapability >= 100 && computeCapability < 120 &&
           acceleratedFeatures;
  }

  bool supportMaximumMinimum() const { return computeCapability >= 80; }

  bool supportLdMatrix() const { return computeCapability >= 75; }
  bool supportStMatrix() const { return computeCapability >= 90; }
  bool supportLdStMatrixB8() const { return computeCapability >= 100; }

  bool supportBitwidth16Elementwise() const {
    // Hopper (sm90) and newer.
    return computeCapability >= 90;
  }

  bool supportBitwidth32Elementwise() const {
    // Blackwell (sm100) and newer.
    return computeCapability >= 100;
  }

private:
  static constexpr char kTargetPrefix[] = "cuda:";

  int computeCapability;
  bool acceleratedFeatures;
};

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TARGETFEATURES_H_
