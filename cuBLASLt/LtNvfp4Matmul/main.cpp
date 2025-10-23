/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cuda_fp4.h>
#include <cuda_bf16.h>

#include "sample_cublasLt_LtNvfp4Matmul.h"
#include "helpers.h"

#include <cstdlib>
#include <cstring> // for strcmp
#include <cassert>

int main(int argc, char** argv) {
    int m = 0, n = 0, k = 0;
    int iters = 1;
    int warmup = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) m = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0) n = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0) k = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0) warmup = std::atoi(argv[++i]);
    }

    if (m == 0 || n == 0 || k == 0) {
        std::cerr << "Usage: " << argv[0] << " -m <M> -n <N> -k <K> [--iters N] [--warmup N]" << std::endl;
        return 1;
    }
    // Block scaling size 16
    /*
    TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3, float, __nv_bfloat16> props(
        CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
    */
    TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3, float, __nv_bfloat16> props(
        CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);
     
    props.run([&props] {
        LtNvfp4Matmul(props.ltHandle,
                    props.transa,
                    props.transb,
                    props.m,
                    props.n,
                    props.k,
                    &props.alpha,
                    props.AscaleDev,
                    props.Adev,
                    props.lda,
                    props.BscaleDev,
                    props.Bdev,
                    props.ldb,
                    &props.beta,
                    props.CscaleDev,
                    props.Cdev,
                    props.ldc,
                    props.DscaleDev,
                    props.Ddev,
                    props.ldd,
                    props.DOutscaleDev,
                    props.workspace,
                    props.workspaceSize,
                    props.AScaleMode,
                    props.BScaleMode,
                    props.CScaleMode,
                    props.DScaleMode,
                    props.DOutScaleMode);
    });

    return 0;
}