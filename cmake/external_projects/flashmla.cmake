include(FetchContent)

set(VENDORED_FLASHMLA_INTERFACE
    "${CMAKE_SOURCE_DIR}/vllm/third_party/flashmla/flash_mla_interface.py")

set(SUPPORT_ARCHS)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3)
    list(APPEND SUPPORT_ARCHS "9.0a")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
    # CUDA 12.9 has introduced "Family-Specific Architecture Features"
    # this supports all compute_10x family
    list(APPEND SUPPORT_ARCHS "10.0f")
elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    list(APPEND SUPPORT_ARCHS "10.0a")
endif()
if(DEFINED ENV{FLASH_MLA_ENABLE_SM70} AND "$ENV{FLASH_MLA_ENABLE_SM70}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
    list(APPEND SUPPORT_ARCHS "7.0")
endif()

cuda_archs_loose_intersection(FLASH_MLA_ARCHS "${SUPPORT_ARCHS}" "${CUDA_ARCHS}")

if(NOT FLASH_MLA_ARCHS AND EXISTS "${VENDORED_FLASHMLA_INTERFACE}")
  message(STATUS "FlashMLA build skipped for CUDA_ARCHS='${CUDA_ARCHS}'; using vendored interface only.")
  return()
endif()

# If FLASH_MLA_SRC_DIR is set, flash-mla is installed from that directory 
# instead of downloading.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{FLASH_MLA_SRC_DIR})
  set(FLASH_MLA_SRC_DIR $ENV{FLASH_MLA_SRC_DIR})
endif()

# Default to the in-tree vendored FlashMLA checkout (./FlashMLA at the repo
# root) when it exists and no explicit FLASH_MLA_SRC_DIR was provided. This lets
# the V100/SM70 fork build the bundled FlashMLA source without a network fetch;
# set FLASH_MLA_SRC_DIR (env or -D) to override.
if(NOT FLASH_MLA_SRC_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/FlashMLA/csrc")
  set(FLASH_MLA_SRC_DIR "${CMAKE_SOURCE_DIR}/FlashMLA")
  message(STATUS "Using in-tree FlashMLA at ${FLASH_MLA_SRC_DIR}")
endif()

if(FLASH_MLA_SRC_DIR)
  FetchContent_Declare(
        flashmla 
        SOURCE_DIR ${FLASH_MLA_SRC_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
else()
  FetchContent_Declare(
        flashmla
        GIT_REPOSITORY https://github.com/vllm-project/FlashMLA
        GIT_TAG a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b
        GIT_PROGRESS TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
  )
endif()


FetchContent_MakeAvailable(flashmla)
message(STATUS "FlashMLA is available at ${flashmla_SOURCE_DIR}")

set(FLASHMLA_HAS_SM70 FALSE)
set(FLASHMLA_HAS_DENSE_EXTENSION_ARCH FALSE)
if("7.0" IN_LIST FLASH_MLA_ARCHS)
    set(FLASHMLA_HAS_SM70 TRUE)
endif()
foreach(_FLASHMLA_ARCH ${FLASH_MLA_ARCHS})
    if("${_FLASHMLA_ARCH}" MATCHES "^(9\\.0a|10\\.0a|10\\.0f)$")
        set(FLASHMLA_HAS_DENSE_EXTENSION_ARCH TRUE)
    endif()
endforeach()

function(_flashmla_get_env_int OUT_VAR ENV_NAME DEFAULT_VALUE ALLOWED_REGEX)
    set(_VALUE "${DEFAULT_VALUE}")
    if(DEFINED ENV{${ENV_NAME}} AND NOT "$ENV{${ENV_NAME}}" STREQUAL "")
        set(_VALUE "$ENV{${ENV_NAME}}")
    endif()
    if(NOT "${_VALUE}" MATCHES "${ALLOWED_REGEX}")
        message(FATAL_ERROR
            "${ENV_NAME} must match ${ALLOWED_REGEX}, got '${_VALUE}'")
    endif()
    set(${OUT_VAR} "${_VALUE}" PARENT_SCOPE)
endfunction()

set(FlashMLA_SM70_COMPILE_DEFINITIONS)
if(FLASHMLA_HAS_SM70)
    _flashmla_get_env_int(FLASHMLA_SM70_H_TILE
        FLASH_MLA_SM70_H_TILE 4 "^(4|8|16)$")
    _flashmla_get_env_int(FLASHMLA_SM70_CTA_THREADS
        FLASH_MLA_SM70_CTA_THREADS 256 "^(128|256)$")
    _flashmla_get_env_int(FLASHMLA_SM70_DENSE_USE_MMA_884_ONLINE
        FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_DENSE_MMA_884_ONLINE_K_TILE
        FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE 0 "^(0|16|32|64)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_CTA_THREADS
        FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS 256 "^(128|256)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_K_TILE
        FLASH_MLA_SM70_SPARSE_DECODE_K_TILE 32 "^(16|32|64)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_QK
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_PV
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_CTA_THREADS
        FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS 256 "^(128|256)$")
    # V100 has 96 KB shared memory per SM; K_TILE=32 uses 38.3 KB/block
    # (2 blocks/SM), K_TILE=16 uses 20 KB/block (4 blocks/SM, 2x occupancy)
    # and measures +37% prefill throughput on DeepSeek V4 Flash. See
    # `.kiro/specs/deepseek-v4-flash-prefill-throughput/`.
    # Allowed values: {16, 32, 64}; env var overrides this default.
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_K_TILE
        FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE 16 "^(16|32|64)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_DOUBLE_BUFFER
        FLASH_MLA_SM70_SPARSE_DECODE_DOUBLE_BUFFER 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_STAGE_Q_HALF
        FLASH_MLA_SM70_SPARSE_DECODE_STAGE_Q_HALF 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF
        FLASH_MLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF 4 "^(0|4|8)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_S0C3_ABLATION
        FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION 0 "^(0|1|2|3|4)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_FAST_E4M3
        FLASH_MLA_SM70_SPARSE_DECODE_FAST_E4M3 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND
        FLASH_MLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_PV_P_CACHE
        FLASH_MLA_SM70_SPARSE_DECODE_PV_P_CACHE 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR
        FLASH_MLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR 0 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_DECODE_BATCH2_VERIFY
        FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY 1 "^(0|1)$")
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER
        FLASH_MLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER 0 "^(0|1)$")
    # Round 4 of deepseek-v4-flash-prefill-throughput spec:
    # batch HEADS_PER_BLOCK heads-of-the-same-query per CUDA block
    # with WARP-SPECIALIZATION so heads execute in parallel and share
    # the KV tile staged once per block. HPB=4 gives 4x KV reuse at
    # 3 CTAs/SM on V100 (from 4 CTAs/SM at HPB=1) and measured
    # +21% prefill tok/s on DeepSeek V4 Flash (93.61 -> 112.89 tok/s
    # on prompt_3k unique; decode unchanged within noise at 19.83).
    # env var FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK overrides.
    # Allowed values: {1, 2, 4, 8}.
    _flashmla_get_env_int(FLASHMLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK
        FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK 4 "^(1|2|4|8)$")

    if(NOT FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE STREQUAL "0")
        set(FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_QK 1)
        set(FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_PV 1)
    endif()
    if(NOT FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE STREQUAL "0")
        set(FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK 1)
        set(FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV 1)
    endif()

    list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS
        KERUTILS_ALLOW_SM70_STUB_COMPILE
        FLASH_MLA_SM70_H_TILE=${FLASHMLA_SM70_H_TILE}
        FLASH_MLA_SM70_CTA_THREADS=${FLASHMLA_SM70_CTA_THREADS}
        FLASH_MLA_SM70_DENSE_USE_MMA_884_ONLINE=${FLASHMLA_SM70_DENSE_USE_MMA_884_ONLINE}
        FLASH_MLA_SM70_DENSE_MMA_884_ONLINE_K_TILE=${FLASHMLA_SM70_DENSE_MMA_884_ONLINE_K_TILE}
        FLASH_MLA_SM70_SPARSE_DECODE_CTA_THREADS=${FLASHMLA_SM70_SPARSE_DECODE_CTA_THREADS}
        FLASH_MLA_SM70_SPARSE_DECODE_K_TILE=${FLASHMLA_SM70_SPARSE_DECODE_K_TILE}
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_QK=${FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_QK}
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_PV=${FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_PV}
        FLASH_MLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE=${FLASHMLA_SM70_SPARSE_DECODE_USE_MMA_884_ONLINE}
        FLASH_MLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF=${FLASHMLA_SM70_SPARSE_DECODE_KV_ROW_PAD_HALF}
        FLASH_MLA_SM70_SPARSE_PREFILL_CTA_THREADS=${FLASHMLA_SM70_SPARSE_PREFILL_CTA_THREADS}
        FLASH_MLA_SM70_SPARSE_PREFILL_K_TILE=${FLASHMLA_SM70_SPARSE_PREFILL_K_TILE}
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK=${FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_QK}
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV=${FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_PV}
        FLASH_MLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE=${FLASHMLA_SM70_SPARSE_PREFILL_USE_MMA_884_ONLINE}
        FLASH_MLA_SM70_SPARSE_DECODE_DOUBLE_BUFFER=${FLASHMLA_SM70_SPARSE_DECODE_DOUBLE_BUFFER}
        FLASH_MLA_SM70_SPARSE_DECODE_STAGE_Q_HALF=${FLASHMLA_SM70_SPARSE_DECODE_STAGE_Q_HALF}
        FLASH_MLA_SM70_SPARSE_DECODE_S0C3_ABLATION=${FLASHMLA_SM70_SPARSE_DECODE_S0C3_ABLATION}
        FLASH_MLA_SM70_SPARSE_DECODE_FAST_E4M3=${FLASHMLA_SM70_SPARSE_DECODE_FAST_E4M3}
        FLASH_MLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND=${FLASHMLA_SM70_SPARSE_DECODE_FAST_BF16_ROUND}
        FLASH_MLA_SM70_SPARSE_DECODE_PV_P_CACHE=${FLASHMLA_SM70_SPARSE_DECODE_PV_P_CACHE}
        FLASH_MLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR=${FLASHMLA_SM70_SPARSE_DECODE_V_FRAGMENT_VECTOR}
        FLASH_MLA_SM70_SPARSE_DECODE_BATCH2_VERIFY=${FLASHMLA_SM70_SPARSE_DECODE_BATCH2_VERIFY}
        FLASH_MLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER=${FLASHMLA_SM70_SPARSE_PREFILL_DOUBLE_BUFFER}
        FLASH_MLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK=${FLASHMLA_SM70_SPARSE_PREFILL_HEADS_PER_BLOCK})

    # Opt-in compile-time metering of WS HPB sparse prefill kernel stages.
    # Set env FLASH_MLA_METER_SPARSE_WS_HPB=1 to enable.
    if(DEFINED ENV{FLASH_MLA_METER_SPARSE_WS_HPB} AND "$ENV{FLASH_MLA_METER_SPARSE_WS_HPB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_SPARSE_WS_HPB)
        message(STATUS "FlashMLA SM70: per-stage metering ENABLED")
    endif()

    # Opt-in FINE-grained (12-slot) metering. Can stack with the 6-slot
    # version; the two sets of counters are independent.
    if(DEFINED ENV{FLASH_MLA_METER_SPARSE_WS_HPB_FINE} AND "$ENV{FLASH_MLA_METER_SPARSE_WS_HPB_FINE}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_SPARSE_WS_HPB_FINE)
        message(STATUS "FlashMLA SM70: per-stage FINE metering ENABLED")
    endif()

    # Opt-in QK sub-stage metering (4 slots inside compute_mma884_qk_group).
    if(DEFINED ENV{FLASH_MLA_METER_QK_SUB} AND "$ENV{FLASH_MLA_METER_QK_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_QK_SUB)
        message(STATUS "FlashMLA SM70: QK sub-stage metering ENABLED")
    endif()

    # Opt-in s4a sub-stage metering (5 slots inside the PV MMA dim_group loop).
    if(DEFINED ENV{FLASH_MLA_METER_S4A_SUB} AND "$ENV{FLASH_MLA_METER_S4A_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_S4A_SUB)
        message(STATUS "FlashMLA SM70: s4a sub-stage metering ENABLED")
    endif()

    # Opt-in decode sparse stage metering (6 stages per tile in the decode kernel).
    if(DEFINED ENV{FLASH_MLA_METER_SPARSE_DECODE} AND "$ENV{FLASH_MLA_METER_SPARSE_DECODE}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_SPARSE_DECODE)
        message(STATUS "FlashMLA SM70: decode sparse stage metering ENABLED")
    endif()

    # Opt-in decode QK sub-stage metering (4 slots inside compute_mma884_qk_group:
    # load_q, load_k, mma884, store).
    if(DEFINED ENV{FLASH_MLA_METER_DECODE_QK_SUB} AND "$ENV{FLASH_MLA_METER_DECODE_QK_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_DECODE_QK_SUB)
        message(STATUS "FlashMLA SM70: decode QK sub-stage metering ENABLED")
    endif()
    if(DEFINED ENV{FLASH_MLA_METER_DECODE_S0_SUB} AND "$ENV{FLASH_MLA_METER_DECODE_S0_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_DECODE_S0_SUB)
        message(STATUS "FlashMLA SM70: decode s0 sub-stage metering ENABLED")
    endif()
    if(DEFINED ENV{FLASH_MLA_METER_DECODE_S0C_SUB} AND "$ENV{FLASH_MLA_METER_DECODE_S0C_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_DECODE_S0C_SUB)
        message(STATUS "FlashMLA SM70: decode s0c sub-stage metering ENABLED")
    endif()
    if(DEFINED ENV{FLASH_MLA_METER_DECODE_S4A_SUB} AND "$ENV{FLASH_MLA_METER_DECODE_S4A_SUB}" MATCHES "^(1|true|TRUE|yes|YES|on|ON)$")
        list(APPEND FlashMLA_SM70_COMPILE_DEFINITIONS FLASH_MLA_METER_DECODE_S4A_SUB)
        message(STATUS "FlashMLA SM70: decode s4/PV sub-stage metering ENABLED")
    endif()
endif()

# Vendor FlashMLA interface into vLLM with torch-ops shim.
set(FLASHMLA_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vllm/third_party/flashmla")
file(MAKE_DIRECTORY "${FLASHMLA_VENDOR_DIR}")
file(READ "${flashmla_SOURCE_DIR}/flash_mla/flash_mla_interface.py"
     FLASHMLA_INTERFACE_CONTENT)
string(REPLACE "import flash_mla.cuda as flash_mla_cuda"
               "import vllm._flashmla_C\nflash_mla_cuda = torch.ops._flashmla_C"
               FLASHMLA_INTERFACE_CONTENT
               "${FLASHMLA_INTERFACE_CONTENT}")
file(WRITE "${FLASHMLA_VENDOR_DIR}/flash_mla_interface.py"
     "${FLASHMLA_INTERFACE_CONTENT}")

# Install the generated flash_mla_interface.py to the wheel
# Use COMPONENT _flashmla_C to ensure it's installed with the C extension
install(FILES "${FLASHMLA_VENDOR_DIR}/flash_mla_interface.py"
        DESTINATION vllm/third_party/flashmla/
        COMPONENT _flashmla_C)

# The FlashMLA kernels only work on hopper and require CUDA 12.3 or later.
# Only build FlashMLA kernels if we are building for something compatible with 
# sm90a
if(FLASH_MLA_ARCHS)
    message(STATUS "FlashMLA CUDA architectures: ${FLASH_MLA_ARCHS}")
    set(VLLM_FLASHMLA_GPU_FLAGS ${VLLM_GPU_FLAGS})
    list(APPEND VLLM_FLASHMLA_GPU_FLAGS "--expt-relaxed-constexpr" "--expt-extended-lambda" "--use_fast_math")

    set(FlashMLA_SM70_SOURCES)
    if(FLASHMLA_HAS_SM70)
        list(APPEND FlashMLA_SM70_SOURCES
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/dense/instantiations/fp16.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/sparse_fp8/instantiations/v32_fp8.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/decode/sparse_fp8/instantiations/model1_fp8.cu
            ${flashmla_SOURCE_DIR}/csrc/sm70/prefill/sparse/instantiations/bf16.cu)
    endif()

    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/torch_api.cpp

        # Misc kernels for decoding
        ${flashmla_SOURCE_DIR}/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
        ${flashmla_SOURCE_DIR}/csrc/smxx/decode/combine/combine.cu
        ${FlashMLA_SM70_SOURCES}

        # sm90 dense decode
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/fp16.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/bf16.cu

        # sm90 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu

        # sm90 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

        # sm100 dense prefill & backward
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu

        # sm100 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu

        # sm100 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/v32.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/model1.cu
        ${flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
    )

    set(FlashMLA_Extension_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/extension/torch_api.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/pybind.cpp
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
    )

    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/kerutils/include
        ${flashmla_SOURCE_DIR}/csrc/sm90
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set(FlashMLA_Extension_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    set_gencode_flags_for_srcs(
        SRCS "${FlashMLA_Extension_SOURCES}"
        CUDA_ARCHS "${FLASH_MLA_ARCHS}")

    define_extension_target(
        _flashmla_C
        DESTINATION vllm
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES ${FlashMLA_SOURCES}
        COMPILE_FLAGS ${VLLM_GPU_FLAGS}
        ARCHITECTURES ${VLLM_GPU_ARCHES}
        INCLUDE_DIRECTORIES ${FlashMLA_INCLUDES}
        USE_SABI 3
        WITH_SOABI)

    # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
    # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
    # Also enable C++20 for the FlashMLA sources (required for std::span, requires, etc.)
    target_compile_options(_flashmla_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
    target_compile_definitions(_flashmla_C PRIVATE
        ${FlashMLA_SM70_COMPILE_DEFINITIONS})

    set(FLASHMLA_BUILD_DENSE_EXTENSION ${FLASHMLA_HAS_DENSE_EXTENSION_ARCH})
    if(FLASHMLA_BUILD_DENSE_EXTENSION)
        define_extension_target(
            _flashmla_extension_C
            DESTINATION vllm
            LANGUAGE ${VLLM_GPU_LANG}
            SOURCES ${FlashMLA_Extension_SOURCES}
            COMPILE_FLAGS ${VLLM_FLASHMLA_GPU_FLAGS}
            ARCHITECTURES ${VLLM_GPU_ARCHES}
            INCLUDE_DIRECTORIES ${FlashMLA_Extension_INCLUDES}
            USE_SABI 3
            WITH_SOABI)

        # Keep Stable ABI for the module, but *not* for CUDA/C++ files.
        # This prevents Py_LIMITED_API from affecting nvcc and C++ compiles.
        target_compile_options(_flashmla_extension_C PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-UPy_LIMITED_API>
            $<$<COMPILE_LANGUAGE:CXX>:-UPy_LIMITED_API>)
    else()
        message(STATUS "FlashMLA dense extension skipped for CUDA architectures: ${FLASH_MLA_ARCHS}")
        add_custom_target(_flashmla_extension_C)
    endif()
else()
    message(STATUS "FlashMLA will not compile: unsupported CUDA architecture ${CUDA_ARCHS}")
    # Create empty targets for setup.py on unsupported systems
    add_custom_target(_flashmla_C)
    add_custom_target(_flashmla_extension_C)
endif()
