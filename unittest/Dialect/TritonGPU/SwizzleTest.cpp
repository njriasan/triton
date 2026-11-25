#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallSet.h>
#include <optional>

using namespace mlir;
using namespace mlir::triton;

using mlir::triton::actionAdditiveStrides;
using mlir::triton::gpu::bankConflictsLdSt;
using mlir::triton::gpu::LocalMemOpTile;
using mlir::triton::gpu::optimalSwizzling;
using mlir::triton::gpu::optimalSwizzlingLdSt;

namespace {

static std::string attrStr(Attribute a) {
  std::string s;
  llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}
class SwizzleTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

class BankConflictTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect>();
  }

  MLIRContext ctx;

  using LinearLayout = mlir::triton::LinearLayout;

  mlir::triton::gpu::BlockedEncodingAttr
  blocked(ArrayRef<unsigned> spt, ArrayRef<unsigned> tpw,
          ArrayRef<unsigned> wpcta, ArrayRef<unsigned> order,
          ArrayRef<unsigned> cpg = {}, ArrayRef<unsigned> split = {},
          ArrayRef<unsigned> cOrder = {}) {
    SmallVector<unsigned> cpgStorage;
    SmallVector<unsigned> splitStorage;
    SmallVector<unsigned> cOrderStorage;
    if (cpg.empty())
      cpgStorage.assign(spt.size(), 1);
    if (split.empty())
      splitStorage.assign(spt.size(), 1);
    if (cOrder.empty())
      cOrderStorage.assign(order.begin(), order.end());

    auto cgaLayout = mlir::triton::gpu::CGAEncodingAttr::fromSplitParams(
        &ctx, cpgStorage.empty() ? cpg : ArrayRef<unsigned>(cpgStorage),
        splitStorage.empty() ? split : ArrayRef<unsigned>(splitStorage),
        cOrderStorage.empty() ? cOrder : ArrayRef<unsigned>(cOrderStorage));
    return mlir::triton::gpu::BlockedEncodingAttr::get(&ctx, spt, tpw, wpcta,
                                                       order, cgaLayout);
  }

  mlir::triton::gpu::NvidiaMmaEncodingAttr mma(ArrayRef<unsigned> version,
                                               ArrayRef<unsigned> warpsPerCTA,
                                               ArrayRef<unsigned> instrShape) {
    auto cgaLayout = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(
        &ctx, warpsPerCTA.size());
    return mlir::triton::gpu::NvidiaMmaEncodingAttr::get(
        &ctx, version[0], version[1], warpsPerCTA, cgaLayout, instrShape);
  }

  mlir::triton::gpu::NVMMASharedEncodingAttr
  nvmmaShared(unsigned swizzle, unsigned bitwidth, unsigned rank,
              bool transposed = false) {
    SmallVector<unsigned> cpg(rank, 1), split(rank, 1), order(rank);
    std::iota(order.begin(), order.end(), 0);
    auto cgaLayout = mlir::triton::gpu::CGAEncodingAttr::fromSplitParams(
        &ctx, cpg, split, order);
    return mlir::triton::gpu::NVMMASharedEncodingAttr::get(
        &ctx, swizzle, transposed, bitwidth,
        /*fp4Padded=*/false, cgaLayout);
  }

  mlir::triton::gpu::AMDMfmaEncodingAttr
  mfma(unsigned version, ArrayRef<unsigned> warpsPerCTA,
       ArrayRef<unsigned> instrShape, bool isTransposed,
       ArrayRef<unsigned> tilesPerWarp = {}, unsigned bitWidth = 0) {
    auto cgaLayout = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(
        &ctx, warpsPerCTA.size());
    return mlir::triton::gpu::AMDMfmaEncodingAttr::get(
        &ctx, version, warpsPerCTA, instrShape, isTransposed, cgaLayout,
        tilesPerWarp, bitWidth);
  }

  mlir::triton::gpu::AMDRotatingSharedEncodingAttr
  AMDRotatingShared(unsigned vec, unsigned perPhase, unsigned maxPhase,
                    ArrayRef<unsigned> order) {
    auto cgaLayout =
        mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, order.size());
    return mlir::triton::gpu::AMDRotatingSharedEncodingAttr::get(
        &ctx, vec, perPhase, maxPhase, order, cgaLayout);
  }

  LinearLayout toLL(ArrayRef<int64_t> shape, Attribute attr) {
    return mlir::triton::gpu::toLinearLayout(shape, attr);
  }

  int computeConflicts(ArrayRef<int64_t> shape, Attribute regAttr,
                       Attribute sharedAttr, int bitwidth, int numBanks = 32,
                       LocalMemOpTile laneTile = {}) {
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);
    return mlir::triton::gpu::bankConflictsMemDesc(regLL, sharedLL, bitwidth,
                                                   numBanks, laneTile);
  }

  int bruteforceBankConflictsPerWavefront(ArrayRef<int64_t> shape,
                                          Attribute regAttr,
                                          Attribute sharedAttr, int bitwidth) {
    // Compute the bank conflicts per wavefront
    // In other words, we compute how many extra memory accesses (bank
    // conflicts) are needed for a given wavefront.
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);

    auto *ctx = sharedLL.getInDimNames().begin()->getContext();
    auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

    auto kOffset = S("offset");
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto regToShared = regLL.invertAndCompose(sharedLL);
    assert(regToShared.isTrivialOver({S("block")}) && "NYI");
    regToShared = regToShared.sublayout({kReg, kLane, kWarp}, {kOffset});

    // Remove broadcasting
    regToShared = actionRemoveBroadcastedRegs(regToShared).apply(regToShared);
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, regToShared, bitwidth);
    regToShared = permutation.apply(regToShared);

    int vectorisation = llvm::divideCeil(bitwidth * elemsPerVec, 32);
    assert(vectorisation == 1 || vectorisation == 2 || vectorisation == 4);
    int wavefronts = 0;
    // For all the emitted instructions
    for (int regIdx = 0; regIdx < regToShared.getInDimSize(kReg);
         regIdx += elemsPerVec) {
      for (int warpIdx = 0; warpIdx < regToShared.getInDimSize(kWarp);
           warpIdx++) {
        // For each instruction
        for (int laneIdx = 0; laneIdx < regToShared.getInDimSize(kLane);
             laneIdx += (32 / vectorisation)) {
          // For each wavefront
          llvm::SmallSet<int, 32> uniqueOffsets;
          for (int laneIdx = 0; laneIdx < 32 / vectorisation; laneIdx++) {
            for (int vecIdx = 0; vecIdx < elemsPerVec; vecIdx++) {
              auto offset = regToShared
                                .apply({{kReg, regIdx + vecIdx},
                                        {kLane, laneIdx},
                                        {kWarp, warpIdx}})[0]
                                .second;
              auto offsetB32 = offset * bitwidth / 32;
              uniqueOffsets.insert(offsetB32);
            }
          }
          llvm::SmallVector<int, 32> banks(32, 0);
          for (int offset : uniqueOffsets) {
            banks[offset % 32]++;
          }
          wavefronts += *llvm::max_element(banks);
        }
      }
    }
    auto minWavefronts =
        (regToShared.getInDimSize(kReg) / elemsPerVec) *
        regToShared.getInDimSize(kWarp) *
        (regToShared.getInDimSize(kLane) / (32 / vectorisation));
    // Assert homogeneity
    assert(wavefronts % minWavefronts == 0);
    return wavefronts / minWavefronts - 1;
  }

  int bruteforceBankConflictsPerWavefront64(ArrayRef<int64_t> shape,
                                            Attribute regAttr,
                                            Attribute sharedAttr, int bitwidth,
                                            int numBanks) {
    // Compute the bank conflicts per wavefront
    // In other words, we compute how many extra memory accesses (bank
    // conflicts) are needed for a given wavefront.
    auto regLL = toLL(shape, regAttr);
    auto sharedLL = toLL(shape, sharedAttr);

    auto *ctx = sharedLL.getInDimNames().begin()->getContext();
    auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };

    auto kOffset = S("offset");
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto regToShared = regLL.invertAndCompose(sharedLL);
    assert(regToShared.isTrivialOver({S("block")}) && "NYI");
    regToShared = regToShared.sublayout({kReg, kLane, kWarp}, {kOffset});

    // Remove broadcasting
    regToShared = actionRemoveBroadcastedRegs(regToShared).apply(regToShared);
    auto [elemsPerVec, permutation] =
        largestVectorisation(ctx, regToShared, bitwidth);
    regToShared = permutation.apply(regToShared);

    int vectorisation = llvm::divideCeil(bitwidth * elemsPerVec, 32);
    assert(vectorisation == 1 || vectorisation == 2 || vectorisation == 4);
    int threadsPerPhase = numBanks / vectorisation;
    int numLanes = regToShared.getInDimSize(kLane);
    int numPhases = numLanes / threadsPerPhase;
    int wavefronts = 0;
    // For all the emitted instructions
    for (int regIdx = 0; regIdx < regToShared.getInDimSize(kReg);
         regIdx += elemsPerVec) {
      for (int warpIdx = 0; warpIdx < regToShared.getInDimSize(kWarp);
           warpIdx++) {
        // For each instruction
        llvm::SmallVector<llvm::SmallSet<int, 32>, 8> uniqueOffsetsByPhase(
            numPhases);
        for (int laneIdx = 0; laneIdx < numLanes; laneIdx++) {
          int phaseIdx;
          if (vectorisation == 4) {
            int t2 = (laneIdx >> 2) & 1;
            int t3 = (laneIdx >> 3) & 1;
            int t4 = (laneIdx >> 4) & 1;
            int t5 = (laneIdx >> 5) & 1;
            if (numBanks == 32) {
              // Group threads into 8 phases of 8 threads each.
              // Logic: 'half' (bit 5) selects even/odd phase indices.
              // Bit 3 (b3) and the XOR of bits 2 and 4 (b2^b4) determine the
              // intra-half grouping.
              // Phase 0: T0-3,   T20-23 | Phase 1: T32-35, T52-55
              // Phase 2: T4-7,   T16-19 | Phase 3: T36-39, T48-51
              // Phase 4: T8-11,  T28-31 | Phase 5: T40-43, T60-63
              // Phase 6: T12-15, T24-27 | Phase 7: T44-47, T56-59
              // phaseIdx is a linear layout on the laneIdx:
              // phase[0] = t[5]
              // phase[1] = t[2] ^ t[4]
              // phase[2] = t[3]
              phaseIdx = (t5 << 0) | ((t2 ^ t4) << 1) | (t3 << 2);
            } else {
              // Group threads into 4 phases of 16 threads each.
              // Logic: 'half' (bit 5) selects Phase 0/1 or 2/3.
              // Parity of bits 2, 3, and 4 (b2^b3^b4) determines the intra-half
              // grouping.
              // Phase 0: T0-3,   T12-15, T20-23, T24-27
              // Phase 1: T32-35, T44-47, T52-55, T56-59
              // Phase 2: T4-7,   T8-11,  T16-19, T28-31
              // Phase 3: T36-39, T40-43, T48-51, T60-63
              // phaseIdx is a linear layout on the laneIdx:
              // phase[0] = t[5]
              // phase[1] = t[2] ^ t[3] ^ t[4]
              phaseIdx = (t5 << 0) | ((t2 ^ t3 ^ t4) << 1);
            }
          } else {
            phaseIdx = laneIdx / threadsPerPhase;
          }
          for (int vecIdx = 0; vecIdx < elemsPerVec; vecIdx++) {
            auto offset = regToShared
                              .apply({{kReg, regIdx + vecIdx},
                                      {kLane, laneIdx},
                                      {kWarp, warpIdx}})[0]
                              .second;
            auto offsetB32 = offset * bitwidth / 32;
            uniqueOffsetsByPhase[phaseIdx].insert(offsetB32);
          }
        }
        for (auto &uniqueOffsets : uniqueOffsetsByPhase) {
          llvm::SmallVector<int, 64> banks(numBanks, 0);
          for (int offset : uniqueOffsets) {
            banks[offset % numBanks]++;
          }
          wavefronts += *llvm::max_element(banks);
        }
      }
    }
    auto minWavefronts = (regToShared.getInDimSize(kReg) / elemsPerVec) *
                         regToShared.getInDimSize(kWarp) * numPhases;
    // Assert homogeneity
    assert(wavefronts % minWavefronts == 0);
    return wavefronts / minWavefronts - 1;
  }

  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

  int computeConflictsByEnumeration(ArrayRef<int64_t> shape, Attribute regAttr,
                                    Attribute sharedAttr, int bitwidth,
                                    int numBanks = 32,
                                    LocalMemOpTile laneTile = {}) {
    return mlir::triton::gpu::bankConflictsMemDescPadded(
        toLL(shape, regAttr), toLL(shape, sharedAttr), /*padIntervals=*/{},
        /*paddings=*/{}, bitwidth, numBanks, laneTile);
  }

  // Worst bank conflicts (extra wavefronts) of accessing the padded shared
  // layout with linear component `smem` from `reg`. Enumerates every access:
  // every register vector, warp and phase of lanes. The lanes of phase 0 are
  // the span of `laneTile.laneMask` (as lane ids) or, if it is empty, the
  // first lanes that fill all the banks. The other phases are its cosets.
  int bruteforcePaddedConflicts(const LinearLayout &reg,
                                const LinearLayout &smem,
                                ArrayRef<unsigned> intervals,
                                ArrayRef<unsigned> paddings, int bitwidth,
                                int numBanks, LocalMemOpTile laneTile = {}) {
    auto kOffset = S("offset");
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto kBlock = S("block");
    auto cvt = reg.invertAndCompose(smem);
    cvt = actionRemoveBroadcastedRegs(cvt).apply(cvt);
    std::optional<int> maxVecElems;
    if (!intervals.empty())
      maxVecElems =
          std::min<int>(*llvm::min_element(intervals), 128 / bitwidth);
    auto [elemsPerVec, perm] =
        largestVectorisation(&ctx, cvt, bitwidth, maxVecElems);
    cvt = perm.apply(cvt);

    int numLanes = cvt.getInDimSize(kLane);
    SmallVector<int> phaseLanes = {0};
    auto addToSpan = [&](int laneId) {
      for (int i = 0, n = phaseLanes.size(); i < n; i++)
        phaseLanes.push_back(phaseLanes[i] ^ laneId);
    };
    if (!laneTile.laneMask.empty()) {
      for (int32_t mask : laneTile.laneMask)
        addToSpan(mask);
    } else {
      int dwordsPerVec = std::max(1, elemsPerVec * bitwidth / 32);
      int lanesPerPhase = std::min(numLanes, numBanks / dwordsPerVec);
      for (int l = 1; l < lanesPerPhase; l *= 2)
        addToSpan(l);
    }

    auto padOffset = [&](int64_t off) {
      int64_t padded = off;
      for (auto [interval, padding] : llvm::zip_equal(intervals, paddings))
        padded += (off / interval) * padding;
      return padded;
    };

    int maxWays = 1;
    for (int r = 0; r < cvt.getInDimSize(kReg); r += elemsPerVec) {
      for (int w = 0; w < cvt.getInDimSize(kWarp); w++) {
        for (int l0 = 0; l0 < numLanes; l0++) {
          llvm::DenseSet<int64_t> dwords;
          for (int p : phaseLanes) {
            for (int v = 0; v < elemsPerVec; v++) {
              auto outs = cvt.apply(
                  {{kReg, r + v}, {kLane, l0 ^ p}, {kWarp, w}, {kBlock, 0}});
              int64_t off = 0;
              for (auto [dim, val] : outs)
                if (dim == kOffset)
                  off = val;
              dwords.insert(padOffset(off) * bitwidth / 32);
            }
          }
          SmallVector<int> ways(numBanks, 0);
          for (int64_t dword : dwords)
            maxWays = std::max(maxWays, ++ways[dword % numBanks]);
        }
      }
    }
    return maxWays - 1;
  }

  // Checks bankConflictsMemDescPadded against the brute force, with and
  // without padding, and the GF(2) analysis without padding.
  void checkPadded(const LinearLayout &reg, const LinearLayout &smem,
                   ArrayRef<unsigned> intervals, ArrayRef<unsigned> paddings,
                   int bitwidth, int numBanks, LocalMemOpTile laneTile = {}) {
    using mlir::triton::gpu::bankConflictsMemDesc;
    using mlir::triton::gpu::bankConflictsMemDescPadded;
    int linear = bruteforcePaddedConflicts(reg, smem, {}, {}, bitwidth,
                                           numBanks, laneTile);
    int padded = bruteforcePaddedConflicts(reg, smem, intervals, paddings,
                                           bitwidth, numBanks, laneTile);
    EXPECT_EQ(bankConflictsMemDesc(reg, smem, bitwidth, numBanks, laneTile),
              linear)
        << reg.invertAndCompose(smem);
    EXPECT_EQ(bankConflictsMemDescPadded(reg, smem, {}, {}, bitwidth, numBanks,
                                         laneTile),
              linear)
        << reg.invertAndCompose(smem);
    EXPECT_EQ(bankConflictsMemDescPadded(reg, smem, intervals, paddings,
                                         bitwidth, numBanks, laneTile),
              padded)
        << reg.invertAndCompose(smem);
    // bankConflictsMemDesc dispatches to the padded analysis.
    EXPECT_EQ(bankConflictsMemDesc(reg, smem, bitwidth, numBanks, laneTile,
                                   intervals, paddings),
              padded)
        << reg.invertAndCompose(smem);
  }
};

// ——— Tests ———

TEST_F(SwizzleTest, AdditiveStridesRejectsTransitiveOverlap) {
  auto kRegister = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  SmallVector<std::pair<StringAttr, int32_t>> outDims = {{kOffset, 2048},
                                                         {kBlock, 1}};

  // The 0x510 basis overlaps the lane address at 0x10, so it must remain in
  // the outer xor offset. The 0x100 basis must remain there too because it
  // overlaps 0x510; moving it into the additive inner loop would incorrectly
  // turn (base ^ 0x510) ^ 0x100 into (base ^ 0x510) + 0x100.
  LinearLayout reps(
      {{kRegister, {{1, 0}, {2, 0}, {4, 0}, {0x100, 0}, {0x510, 0}}},
       {kLane, {{0x10, 0}}},
       {kWarp, {}},
       {kBlock, {}}},
      outDims, /*requireSurjective=*/false);
  LinearLayout addrLayout({{kLane, {{0x10, 0}}}, {kWarp, {}}, {kBlock, {}}},
                          outDims, /*requireSurjective=*/false);

  auto result = actionAdditiveStrides(reps, addrLayout, /*maskSpanOffsets=*/0,
                                      /*maskSpanBlocks=*/0,
                                      /*regsPerInst=*/8);
  EXPECT_EQ(result.first, 8);
}

TEST_F(SwizzleTest, AdditiveStridesKeepsIndependentRegisterBases) {
  auto kRegister = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  SmallVector<std::pair<StringAttr, int32_t>> outDims = {{kOffset, 2048},
                                                         {kBlock, 1}};

  LinearLayout reps(
      {{kRegister, {{1, 0}, {2, 0}, {4, 0}, {0x100, 0}, {0x500, 0}}},
       {kLane, {{0x10, 0}}},
       {kWarp, {}},
       {kBlock, {}}},
      outDims, /*requireSurjective=*/false);
  LinearLayout addrLayout({{kLane, {{0x10, 0}}}, {kWarp, {}}, {kBlock, {}}},
                          outDims, /*requireSurjective=*/false);

  auto result = actionAdditiveStrides(reps, addrLayout, /*maskSpanOffsets=*/0,
                                      /*maskSpanBlocks=*/0,
                                      /*regsPerInst=*/8);
  EXPECT_EQ(result.first, 32);
}

TEST_F(SwizzleTest, Test128x128Float8Transpose) {
  // 128x128 float8 matrix transpose
  LinearLayout matrix(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}},
       {S("lane"), {{0, 16}, {0, 32}, {0, 64}, {4, 0}, {8, 0}}},
       {S("warp"), {{16, 0}, {32, 0}, {64, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}}, /*requireSurjective=*/true);
  auto matrix_t = transposeLinearLayout(matrix, {1, 0});

  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/8);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/8);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test16x16Bf16BlockedMma) {
  // 16×16 bf16 MMA
  LinearLayout blocked({{S("register"), {{0, 1}, {0, 2}, {0, 4}}},
                        {S("lane"), {{0, 8}, {1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                        {S("warp"), {}},
                        {S("block"), {}}},
                       {{S("dim0"), 16}, {S("dim1"), 16}},
                       /*requireSurjective=*/true);
  LinearLayout mma({{S("register"), {{0, 1}, {8, 0}, {0, 8}}},
                    {S("lane"), {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}}},
                    {S("warp"), {}},
                    {S("block"), {}}},
                   {{S("dim0"), 16}, {S("dim1"), 16}},
                   /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/16);
  auto [r, w] = bankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(BankConflictTest, F64MmaV2BSharedLayout) {
  using mlir::triton::gpu::DotOperandEncodingAttr;
  using mlir::triton::gpu::SwizzledSharedEncodingAttr;

  SmallVector<int64_t> shape = {4, 16};
  auto src = blocked(/*spt=*/{1, 1}, /*tpw=*/{2, 16}, /*wpcta=*/{4, 1},
                     /*order=*/{1, 0});
  auto mmaV2 = mma({2, 0}, {4, 1}, {8, 8});
  auto dst = DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2,
                                         /*kWidth=*/1);
  auto cgaLayout = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2);
  auto shared = SwizzledSharedEncodingAttr::get(
      &ctx, dst, shape, /*order=*/{1, 0}, cgaLayout, /*typeWidthInBit=*/64);
  EXPECT_EQ(shared.getVec(), 4);
  EXPECT_EQ(shared.getPerPhase(), 1);
  EXPECT_EQ(shared.getMaxPhase(), 4);
  EXPECT_EQ(computeConflicts(shape, src, shared, /*bitwidth=*/64), 0);
  EXPECT_EQ(computeConflicts(shape, dst, shared, /*bitwidth=*/64), 0);

  auto srcLL =
      actionRemoveBroadcastedRegs(toLL(shape, src)).apply(toLL(shape, src));
  auto dstLL =
      actionRemoveBroadcastedRegs(toLL(shape, dst)).apply(toLL(shape, dst));
  auto optimal = optimalSwizzlingLdSt(srcLL, dstLL, /*bitwidth=*/64);
  auto [readConflicts, writeConflicts] =
      bankConflictsLdSt(srcLL, dstLL, optimal, /*bitwidth=*/64);
  EXPECT_EQ(readConflicts, 0);
  EXPECT_EQ(writeConflicts, 0);
}

TEST_F(SwizzleTest, Test16x256U4Mma) {
  // 16×256 u4 MMA
  LinearLayout blocked(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {4, 0}, {8, 0}}},
       {S("lane"), {{0, 32}, {0, 64}, {0, 128}, {1, 0}, {2, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);
  LinearLayout mma(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {8, 0}, {0, 32}, {0, 64}, {0, 128}}},
       {S("lane"), {{0, 8}, {0, 16}, {1, 0}, {2, 0}, {4, 0}}},
       {S("warp"), {}},
       {S("block"), {}}},
      {{S("dim0"), 16}, {S("dim1"), 256}}, /*requireSurjective=*/true);

  auto smem = optimalSwizzlingLdSt(blocked, mma, /*bitwidth=*/4);
  auto [r, w] = bankConflictsLdSt(blocked, mma, smem, /*bitwidth=*/4);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test32x16F32Transpose) {
  // 32×16 f32 transpose
  LinearLayout matrix({{S("register"), {{4, 0}, {8, 0}, {16, 0}}},
                       {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}}},
                       {S("warp"), {{2, 0}}},
                       {S("block"), {}}},
                      {{S("dim0"), 32}, {S("dim1"), 16}},
                      /*requireSurjective=*/true);
  LinearLayout matrix_t({{S("register"), {{0, 2}, {0, 4}, {0, 8}}},
                         {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                         {S("warp"), {{0, 1}}},
                         {S("block"), {}}},
                        {{S("dim0"), 32}, {S("dim1"), 16}},
                        /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/32);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/32);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test128x128F16Transpose) {
  LinearLayout matrix(
      {{S("register"), {{1, 0}, {2, 0}, {4, 0}, {0, 32}, {0, 64}}},
       {S("lane"), {{8, 0}, {16, 0}, {32, 0}, {64, 0}, {0, 1}}},
       {S("warp"), {{0, 2}, {0, 4}, {0, 8}, {0, 16}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout matrix_t(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {32, 0}, {64, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}}},
       {S("warp"), {{2, 0}, {4, 0}, {8, 0}, {16, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 128}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(matrix, matrix_t, /*bitwidth=*/16);
  auto [r, w] = bankConflictsLdSt(matrix, matrix_t, smem, /*bitwidth=*/16);
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test64x128F16BlockedLinear32Bank) {
  LinearLayout src(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}, {2, 0}}},
       {S("warp"), {{4, 0}, {8, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout dst(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 8}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 4}}},
       {S("warp"), {{0, 0}, {0, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/16, /*numBanks*/ 32,
                                   /*srcTile*/ {},
                                   /*dstTile*/ {{}, {}, {1, 2, 20}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/16,
                                  /*numBanks*/ 32, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {}, {1, 2, 20}});
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test64x128F16BlockedMfma64Bank) {
  LinearLayout src(
      {{S("register"), {{0, 1}, {0, 2}, {0, 4}, {16, 0}, {32, 0}}},
       {S("lane"), {{0, 8}, {0, 16}, {0, 32}, {0, 64}, {1, 0}, {2, 0}}},
       {S("warp"), {{4, 0}, {8, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  LinearLayout dst(
      {{S("register"),
        {{0, 1}, {0, 2}, {0, 4}, {0, 16}, {0, 32}, {0, 64}, {32, 0}}},
       {S("lane"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}, {0, 8}}},
       {S("warp"), {{0, 0}, {0, 0}}},
       {S("block"), {}}},
      {{S("dim0"), 64}, {S("dim1"), 128}},
      /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/16,
                                   /*numBanks*/ 64, /*srcTile*/ {},
                                   /*dstTile*/ {{}, {}, {1, 2, 12, 20}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/16,
                                  /*numBanks*/ 64, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {}, {1, 2, 12, 20}});
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test1024F32WarpSwapped32Bank) {
  LinearLayout src({{S("register"), {{1}, {2}}},
                    {S("lane"), {{4}, {8}, {16}, {32}, {64}, {128}}},
                    {S("warp"), {{256}, {512}}},
                    {S("block"), {}}},
                   {{S("dim0"), 1024}},
                   /*requireSurjective=*/true);
  LinearLayout dst({{S("register"), {{1}, {2}}},
                    {S("lane"), {{4}, {8}, {16}, {32}, {64}, {128}}},
                    {S("warp"), {{512}, {256}}},
                    {S("block"), {}}},
                   {{S("dim0"), 1024}},
                   /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/32,
                                   /*numBanks*/ 32, /*srcTile*/ {},
                                   /*dstTile*/ {{}, {}, {1, 2, 20}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/32,
                                  /*numBanks*/ 32, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {}, {1, 2, 20}});
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(SwizzleTest, Test1024F32WarpSwapped64Bank) {
  LinearLayout src({{S("register"), {{1}, {2}}},
                    {S("lane"), {{4}, {8}, {16}, {32}, {64}, {128}}},
                    {S("warp"), {{256}, {512}}},
                    {S("block"), {}}},
                   {{S("dim0"), 1024}},
                   /*requireSurjective=*/true);
  LinearLayout dst({{S("register"), {{1}, {2}}},
                    {S("lane"), {{4}, {8}, {16}, {32}, {64}, {128}}},
                    {S("warp"), {{512}, {256}}},
                    {S("block"), {}}},
                   {{S("dim0"), 1024}},
                   /*requireSurjective=*/true);
  auto smem = optimalSwizzlingLdSt(src, dst, /*bitwidth=*/32,
                                   /*numBanks*/ 64, /*srcTile*/ {},
                                   /*dstTile*/ {{}, {}, {1, 2, 12, 20}});
  auto [r, w] = bankConflictsLdSt(src, dst, smem, /*bitwidth=*/32,
                                  /*numBanks*/ 64, /*srcTile*/ {},
                                  /*dstTile*/ {{}, {}, {1, 2, 12, 20}});
  EXPECT_EQ(r, 0);
  EXPECT_EQ(w, 0);
}

TEST_F(BankConflictTest, bankConflicts) {
  using mlir::triton::gpu::DotOperandEncodingAttr;

  auto mmaV3 = mma({3, 0}, {4, 1}, {16, 32, 16});
  auto mmaV2 = mma({2, 0}, {1, 4}, {16, 8});
  auto mmaV2Large = mma({2, 0}, {2, 4}, {16, 8});

  auto dotA =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV2, /*kWidth=*/2);
  auto dotB =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2, /*kWidth=*/2);
  auto dotBInt8 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2, /*kWidth=*/1);
  auto dotBNoswizzle =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV2Large, /*kWidth=*/2);

  struct Case {
    Attribute reg;
    Attribute shared;
    SmallVector<int64_t, 3> shape;
    int bitwidth;
  };

  SmallVector<Case, 11> cases = {
      {blocked({1}, {32}, {4}, {0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 1, 1, 1, {0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 1)),
       {32},
       32},
      {blocked({1}, {32}, {4}, {0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 1, 1, 1, {0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 1)),
       {32},
       16},
      {mmaV3,
       nvmmaShared(/*swizzle=*/128, /*bitwidth=*/16, /*rank=*/2),
       {128, 128},
       16},
      {dotB,
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2),
       {64, 32},
       16},
      {dotA,
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2,
                   /*transposed=*/true),
       {32, 64},
       16},
      {dotBInt8,
       nvmmaShared(/*swizzle=*/32, /*bitwidth=*/8, /*rank=*/2),
       {8, 32},
       8},
      {mma({2, 0}, {4, 1}, {16, 8}),
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2,
                   /*transposed=*/true),
       {64, 64},
       16},
      {mma({3, 0}, {2, 2}, {16, 32, 16}),
       nvmmaShared(/*swizzle=*/64, /*bitwidth=*/16, /*rank=*/2),
       {64, 32},
       16},
      {mma({2, 0}, {4, 1}, {16, 8}),
       nvmmaShared(/*swizzle=*/32, /*bitwidth=*/8, /*rank=*/2),
       {32, 32},
       8},
      {dotBNoswizzle,
       nvmmaShared(/*swizzle=*/0, /*bitwidth=*/16, /*rank=*/2),
       {4, 64},
       16},
      {mma({3, 0}, {4, 1}, {16, 32, 16}),
       nvmmaShared(/*swizzle=*/128, /*bitwidth=*/32, /*rank=*/2),
       {128, 64},
       32},
  };

  for (const auto &c : cases) {
    EXPECT_EQ(
        computeConflictsByEnumeration(c.shape, c.reg, c.shared, c.bitwidth),
        computeConflicts(c.shape, c.reg, c.shared, c.bitwidth))
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
    EXPECT_EQ(computeConflicts(c.shape, c.reg, c.shared, c.bitwidth),
              bruteforceBankConflictsPerWavefront(c.shape, c.reg, c.shared,
                                                  c.bitwidth))

        << toLL(c.shape, c.reg).invertAndCompose(toLL(c.shape, c.shared))
        << "\nbitwidth=" << c.bitwidth << "\n"
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
  }
}

TEST_F(BankConflictTest, bankConflictsWavefront64) {
  using mlir::triton::gpu::DotOperandEncodingAttr;

  auto mmaV3 = mfma(3, {4, 1}, {32, 32, 8}, true);
  auto mmaV4 = mfma(4, {4, 1}, {32, 32, 16}, true);

  auto dotAV3 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV3, /*kWidth=*/4);
  auto dotAV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV4, /*kWidth=*/8);
  auto dotBV3 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV3, /*kWidth=*/4);
  auto dotBV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV4, /*kWidth=*/4);

  struct Case {
    Attribute reg;
    Attribute shared;
    SmallVector<int64_t, 3> shape;
    int bitwidth;
    int numBanks;
    LocalMemOpTile laneTile;
  };

  SmallVector<Case, 6> cases = {
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       32,
       /*vec=4*/ {{}, {}, {1, 2, 20}}},
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       64,
       /*vec=4*/ {{}, {}, {1, 2, 12, 20}}},
      {dotAV3,
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 4, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       32,
       /*vec=2*/ {{}, {}}},
      {dotAV4,
       mlir::triton::gpu::SwizzledSharedEncodingAttr::get(
           &ctx, 8, 1, 16, {1, 0},
           mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2)),
       {128, 128},
       16,
       64,
       /*vec=4*/ {{}, {}, {1, 2, 12, 20}}},
      {dotBV3,
       AMDRotatingShared(/*vec=*/4, /*perPhase=*/1, /*maxPhase=*/16,
                         /*order=*/{0, 1}),
       {64, 128},
       16,
       32,
       /*vec=2*/ {{}, {}}},
      {dotBV4,
       AMDRotatingShared(/*vec=*/4, /*perPhase=*/2, /*maxPhase=*/8,
                         /*order=*/{0, 1}),
       {64, 128},
       16,
       64,
       /*vec=2*/ {{}, {}}},
  };

  for (const auto &c : cases) {
    EXPECT_EQ(computeConflictsByEnumeration(c.shape, c.reg, c.shared,
                                            c.bitwidth, c.numBanks, c.laneTile),
              computeConflicts(c.shape, c.reg, c.shared, c.bitwidth, c.numBanks,
                               c.laneTile))
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
    EXPECT_EQ(computeConflicts(c.shape, c.reg, c.shared, c.bitwidth, c.numBanks,
                               c.laneTile),
              bruteforceBankConflictsPerWavefront64(c.shape, c.reg, c.shared,
                                                    c.bitwidth, c.numBanks))
        << toLL(c.shape, c.reg).invertAndCompose(toLL(c.shape, c.shared))
        << "\nbitwidth=" << c.bitwidth << "\n"
        << "numBanks=" << c.numBanks << "\n"
        << attrStr(c.reg) << "\n"
        << attrStr(c.shared);
  }
}

// The A operand of an MXFP6 interwave kernel on gfx950 uses
// #ttg.padded_shared<[1024:+16]> over bytes. Each lane reads 32 contiguous
// bytes, from byte ((t<<1)&352)|((t<<7)&8320)|((t<<9)&7168) for thread t.
// Ignoring the padding overestimates the conflicts; the padded counts match
// SQ_LDS_BANK_CONFLICT measurements on MI350X.
TEST_F(BankConflictTest, paddedMxfp6AOperand) {
  using mlir::triton::gpu::bankConflictsMemDesc;
  using mlir::triton::gpu::bankConflictsMemDescPadded;
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kDim = S("dim0");
  auto smem = LinearLayout::identity1D(16384, S("offset"), kDim) *
              LinearLayout::identity1D(1, kBlock, kDim);
  SmallVector<unsigned> intervals = {1024};
  SmallVector<unsigned> paddings = {16};
  std::vector<std::vector<int32_t>> lane = {{128},  {1024}, {2048},
                                            {4096}, {32},   {64}};

  // ds_read_b64 with phases of 32 sequential lanes. Byte 8 of each lane is
  // read by another warp so that the reads do not vectorise to b128.
  auto regB64 = LinearLayout({{kReg, {{1}, {2}, {4}, {16}, {512}}},
                              {kLane, lane},
                              {kWarp, {{8192}, {256}, {8}}},
                              {kBlock, {}}},
                             {kDim});
  // 8-way ignoring the padding, 2-way with it.
  EXPECT_EQ(bankConflictsMemDesc(regB64, smem, /*bitwidth=*/8,
                                 /*numBanks=*/64),
            7);
  EXPECT_EQ(bankConflictsMemDescPadded(regB64, smem, intervals, paddings,
                                       /*bitwidth=*/8, /*numBanks=*/64),
            1);
  checkPadded(regB64, smem, intervals, paddings, /*bitwidth=*/8,
              /*numBanks=*/64);

  // ds_read_b128 with the CDNA4 phases.
  LocalMemOpTile cdna4B128 = {{}, {}, {1, 2, 12, 20}};
  auto regB128 = LinearLayout({{kReg, {{1}, {2}, {4}, {8}, {16}, {512}}},
                               {kLane, lane},
                               {kWarp, {{8192}, {256}}},
                               {kBlock, {}}},
                              {kDim});
  // 4-way ignoring the padding, 2-way with it.
  EXPECT_EQ(bankConflictsMemDesc(regB128, smem, /*bitwidth=*/8,
                                 /*numBanks=*/64, cdna4B128),
            3);
  EXPECT_EQ(bankConflictsMemDescPadded(regB128, smem, intervals, paddings,
                                       /*bitwidth=*/8, /*numBanks=*/64,
                                       cdna4B128),
            1);
  checkPadded(regB128, smem, intervals, paddings, /*bitwidth=*/8,
              /*numBanks=*/64, cdna4B128);

  // Control: lane t reads 16 bytes at t * 16. Conflict free.
  auto smemCtrl = LinearLayout::identity1D(4096, S("offset"), kDim) *
                  LinearLayout::identity1D(1, kBlock, kDim);
  auto regCtrl = LinearLayout({{kReg, {{1}, {2}, {4}, {8}}},
                               {kLane, {{16}, {32}, {64}, {128}, {256}, {512}}},
                               {kWarp, {{1024}, {2048}}},
                               {kBlock, {}}},
                              {kDim});
  EXPECT_EQ(bankConflictsMemDescPadded(regCtrl, smemCtrl, intervals, paddings,
                                       /*bitwidth=*/8, /*numBanks=*/64,
                                       cdna4B128),
            0);
  checkPadded(regCtrl, smemCtrl, intervals, paddings, /*bitwidth=*/8,
              /*numBanks=*/64, cdna4B128);
}

// Classic +1 padding: the lanes read a column of a row-major 32x32 fp32 tile.
TEST_F(BankConflictTest, paddedColumnAccess) {
  using mlir::triton::gpu::bankConflictsMemDesc;
  using mlir::triton::gpu::bankConflictsMemDescPadded;
  auto kDim = S("dim0");
  auto kBlock = S("block");
  auto smem = LinearLayout::identity1D(2048, S("offset"), kDim) *
              LinearLayout::identity1D(1, kBlock, kDim);
  // Lane i reads row i. The warps hold the columns, so the reads are scalar.
  auto reg = LinearLayout({{S("register"), {{1024}}},
                           {S("lane"), {{32}, {64}, {128}, {256}, {512}}},
                           {S("warp"), {{1}, {2}, {4}, {8}, {16}}},
                           {kBlock, {}}},
                          {kDim});
  // Without padding all the rows start in the same bank.
  EXPECT_EQ(bankConflictsMemDesc(reg, smem, /*bitwidth=*/32), 31);
  // [32:+1] moves each row by one bank.
  EXPECT_EQ(bankConflictsMemDescPadded(reg, smem, {32}, {1}, /*bitwidth=*/32),
            0);
  checkPadded(reg, smem, {32}, {1}, /*bitwidth=*/32, /*numBanks=*/32);
  checkPadded(reg, smem, {32}, {2}, /*bitwidth=*/32, /*numBanks=*/32);
  checkPadded(reg, smem, {64}, {1}, /*bitwidth=*/32, /*numBanks=*/32);
}

// Multi-tier padding, which composes additively, and padding on top of a
// swizzled linear component.
TEST_F(BankConflictTest, paddedMultiTierAndSwizzle) {
  using mlir::triton::gpu::DotOperandEncodingAttr;
  using mlir::triton::gpu::SwizzledSharedEncodingAttr;
  auto cga2 = mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, 2);
  auto mmaV4 = mfma(4, {4, 1}, {32, 32, 16}, true);
  auto dotAV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/0, mmaV4, /*kWidth=*/8);
  auto dotBV4 =
      DotOperandEncodingAttr::get(&ctx, /*opIdx=*/1, mmaV4, /*kWidth=*/8);
  LocalMemOpTile cdna4B128 = {{}, {}, {1, 2, 12, 20}};

  struct Case {
    Attribute reg;
    Attribute shared;
    SmallVector<int64_t, 2> shape;
    int bitwidth;
    SmallVector<unsigned> intervals;
    SmallVector<unsigned> paddings;
    int numBanks;
    LocalMemOpTile laneTile;
  };
  auto identity = SwizzledSharedEncodingAttr::get(&ctx, 1, 1, 1, {1, 0}, cga2);
  auto swizzled = SwizzledSharedEncodingAttr::get(&ctx, 8, 1, 8, {1, 0}, cga2);
  SmallVector<Case, 8> cases = {
      // Multi-tier padding.
      {blocked({1, 2}, {16, 4}, {4, 1}, {1, 0}),
       identity,
       {64, 64},
       32,
       {2, 4},
       {1, 2},
       64,
       {}},
      {blocked({1, 1}, {64, 1}, {1, 4}, {1, 0}),
       identity,
       {64, 64},
       32,
       {64, 256},
       {1, 4},
       64,
       {}},
      {blocked({4, 1}, {16, 4}, {1, 4}, {0, 1}),
       identity,
       {128, 64},
       16,
       {64, 512},
       {4, 8},
       32,
       {}},
      {dotAV4, identity, {128, 128}, 16, {128, 1024}, {8, 16}, 64, cdna4B128},
      // Swizzle plus padding.
      {dotAV4, swizzled, {128, 128}, 16, {512}, {8}, 64, cdna4B128},
      {dotBV4, swizzled, {128, 128}, 16, {256, 1024}, {8, 4}, 64, cdna4B128},
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       swizzled,
       {128, 128},
       16,
       {128},
       {8},
       64,
       cdna4B128},
      {blocked({1, 8}, {4, 16}, {4, 1}, {1, 0}),
       swizzled,
       {128, 128},
       16,
       {64, 2048},
       {2, 8},
       32,
       {}},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(attrStr(c.reg) + "\n" + attrStr(c.shared));
    auto reg = toLL(c.shape, c.reg);
    auto smem = toLL(c.shape, c.shared);
    // The CDNA4 tile only describes ds_read_b128, like getSharedLdStTiles.
    auto laneTile = c.laneTile;
    auto cvt = reg.invertAndCompose(smem);
    auto [elemsPerVec, perm] = largestVectorisation(
        &ctx, cvt, c.bitwidth,
        std::min<int>(*llvm::min_element(c.intervals), 128 / c.bitwidth));
    if (elemsPerVec * c.bitwidth != 128)
      laneTile = {};
    checkPadded(reg, smem, c.intervals, c.paddings, c.bitwidth, c.numBanks,
                laneTile);
  }
}

} // namespace

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
