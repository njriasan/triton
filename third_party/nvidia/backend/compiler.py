from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional, Union
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        elif lhs_bitwidth == 64:
            return (1, 1, 4)
        elif lhs_bitwidth == 32:
            return (1, 1, 8)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version(arch: int = 80):
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas(arch).path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas(arch).version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm 979132a0.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(90, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class NvidiaTargetDescriptor:
    arch: str
    capability: int
    suffix: str = ""

    @property
    def llvm_arch(self) -> str:
        return f"sm_{self.capability}{self.suffix}"

    @property
    def ttg_target(self) -> str:
        return f"cuda:{self.arch}"

    @property
    def has_accelerated_features(self) -> bool:
        return self.suffix == "a"


def _normalize_cuda_arch(arch: Union[int, str], *, allow_ambiguous_sm90_plus: bool) -> str:
    if isinstance(arch, int):
        if arch >= 90 and not allow_ambiguous_sm90_plus:
            raise ValueError(f"CUDA target sm{arch} is ambiguous. Use an explicit target string, "
                             f"e.g. sm{arch} or sm{arch}a.")
        return f"sm{arch}"

    if not isinstance(arch, str):
        raise TypeError(f"CUDA target architecture must be an int or str, got {type(arch).__name__}")

    if arch.startswith("cuda:"):
        arch = arch.split(":", 1)[1]

    if arch.startswith("sm_"):
        arch = "sm" + arch[3:]

    # Keep accepting legacy numeric strings below SM90 for compatibility with
    # paths that deserialize old metadata.
    if re.fullmatch(r"\d+", arch):
        capability = int(arch)
        if capability >= 90 and not allow_ambiguous_sm90_plus:
            raise ValueError(f"CUDA target sm{capability} is ambiguous. Use an explicit target string, "
                             f"e.g. sm{capability} or sm{capability}a.")
        return f"sm{capability}"

    pattern = r"^sm(\d+)(a?)$"
    match = re.fullmatch(pattern, arch)
    if not match:
        raise ValueError(f"CUDA target architecture must have the form {pattern}")

    capability = int(match.group(1))
    suffix = match.group(2)
    if suffix and capability < 90:
        raise ValueError(f"CUDA target {arch} uses an 'a' suffix before SM90")
    return f"sm{capability}{suffix}"


def make_cuda_arch(arch: Union[int, str], *, allow_ambiguous_sm90_plus: bool = False) -> NvidiaTargetDescriptor:
    normalized = _normalize_cuda_arch(arch, allow_ambiguous_sm90_plus=allow_ambiguous_sm90_plus)
    match = re.fullmatch(r"^sm(\d+)(a?)$", normalized)
    assert match is not None
    return NvidiaTargetDescriptor(arch=normalized, capability=int(match.group(1)), suffix=match.group(2))


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    ptx_version: int = None
    ptx_options: Optional[str] = knobs.nvidia.ptxas_options
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    enable_reflect_ftz: bool = True  # ftz in libdevice
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee", 'bf16x3', 'bf16x6')
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')
        if "gsan" in self.instrumentation_mode:
            gsan_lib = default_libdir / "gsan.ll"
            if not gsan_lib.exists():
                raise FileNotFoundError(f"GSan runtime is missing at {gsan_lib}. "
                                        "Rebuild Triton to generate it.")
            extern_libs['gsan'] = str(gsan_lib)

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @property
    def enable_iisan(self):
        return "iisan" in self.instrumentation_mode


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        return make_cuda_arch(arch)

    def get_target_name(self, options) -> str:
        target = self._parse_arch(options.arch)
        return target.ttg_target

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        # Enable debug mode for ConSan, so device-side assertions are not optimized out
        if any(mode in opts.get("instrumentation_mode", "") for mode in ["consan", "iisan"]):
            opts["debug"] = True
            opts["sanitize_overflow"] = False

        if knobs.runtime.override_arch:
            arch = knobs.runtime.override_arch
        else:
            arch = self.target.arch
        args = {'arch': arch}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        target = self._parse_arch(args["arch"])
        capability = target.capability
        args["arch"] = target.arch

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = self._parse_arch(options.arch).capability
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        target = make_cuda_arch(opt.arch)
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emuTF32 = (capability // 10 >= 8)
        passes.ttir.add_convert_to_ttgpuir(pm, target.ttg_target, opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_optimize_partition_warps(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 == 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        if "fpsan" in opt.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'make_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_infer_coalesced_encodings(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        if "fpsan" in options.instrumentation_mode:
            passes.ttgpuir.add_fp_sanitizer(pm)
        if any(mode in options.instrumentation_mode for mode in ["consan", "fpsan"]):
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)

        pm.run(mod, 'gluon_to_ttgir')
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        target = make_cuda_arch(options.arch)
        ptx_version = get_ptx_version_from_options(options, target.capability)

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        if "gsan" in options.instrumentation_mode:
            # GSan introduces layout conversions, so must come before shared memory allocation
            passes.ttgpuir.add_global_sanitizer(pm)

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm, "consan" in options.instrumentation_mode)
        passes.convert.add_scf_to_cf(pm)
        passes.gluon.add_inliner(pm)
        if "consan" in options.instrumentation_mode:
            passes.ttgpuir.add_prepare_consan_captures(pm, "nvidia")
        nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_tmem_barrier_insertion(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version, "consan" in options.instrumentation_mode)
        nvidia.passes.ttnvgpuir.add_initialize_ws_cluster_barriers(pm, capability, ptx_version)
        passes.ttgpuir.add_canonicalize_llvm_ir(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        pm.run(mod, 'make_llir')

        if knobs.compilation.dump_ir_extract_di_local_variables:
            # comments below on why separate it
            if not knobs.compilation.disable_line_info:
                pm = ir.pass_manager(mod.context)
                pm.enable_debug()
                passes.llvmir.add_di_scope(pm)
                pm.run(mod, 'make_llir.disable_line_info')

            # insert dbg intrinsic with several DI Attribute including source
            # var name and type info note: unknown reason for now, but this
            # pass and add_di_scope has to be run separately, otherwise if we
            # put them into previous pipline, it trigger a segmentfault without
            # any error message; could be due to a bug in mlir or pybind11
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_local_variable(pm)
            pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        proc = target.llvm_arch
        features = get_features(options, target.capability)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        if options.enable_reflect_ftz:
            nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        # Work around ptxas rejecting PTX generated by the LLVM SLP vectorizer on sm_80.
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3, disable_slp_vectorizer=capability == 80)

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_ptx(self, src, metadata, opt, capability):
        target = make_cuda_arch(opt.arch)
        ptx_version = get_ptx_version_from_options(opt, target.capability)

        triple = 'nvptx64-nvidia-cuda'
        proc = target.llvm_arch
        features = get_features(opt, target.capability)
        flags = ["nvptx-mad-wide-opt"]
        canonicalize_gep = "fpsan" in opt.instrumentation_mode
        ret = llvm.translate_to_asm(src, triple, proc, features, flags, opt.enable_fp_fusion, False, canonicalize_gep)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+a?', f'.target {target.llvm_arch}', ret, flags=re.MULTILINE)
        if not knobs.compilation.dump_ir_extract_di_local_variables:
            # Remove the debug flag that prevents ptxas from optimizing the code
            # Note: if this flag is removed, the source var name and type info will be lost when ptx was compiled into cubin
            #           and we may not be able to see them in cuda-gdb
            ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if knobs.nvidia.dump_nvptx:
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        target = make_cuda_arch(opt.arch)
        ptxas = get_ptxas(target.capability).path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = target.llvm_arch

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            # -Ofc mid miscompiles some large ConSan kernels into invalid global
            # accesses; -O1 keeps compile time reasonable without that ptxas bug.
            if (not knobs.nvidia.disable_ptxas_opt
                    and any(mode in opt.instrumentation_mode for mode in ["consan", "fpsan"])):
                ptx_extra_options += ["--opt-level", "1"]

            # Add --regAllocOptLevel=2 to work around ptxas 13.x bug
            reg_alloc = ['--regAllocOptLevel=2']

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *reg_alloc, *ptx_extra_options, f'--gpu-name={arch}',
                fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options, language):
        target = self._parse_arch(options.arch)
        capability = target.capability
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, capability)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    @functools.lru_cache()
    def hash(self):
        target = make_cuda_arch(self.target.arch)
        version = get_ptxas_version(target.capability)
        return f'{version}-{target.arch}'
