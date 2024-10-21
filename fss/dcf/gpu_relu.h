// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "gpu_dcf.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_select.h"

namespace dcf
{
    template <typename T>
    struct GPUSelectExtKey{
        int N;
        T *rm, *rmd, *rmu, *ud, *m, *v, *w, *z, *rin;
    };
    template <typename T>
    struct GPUReluExtKey
    {
        int bin, bout, N;
        dpf::GPUDReluKey dReluKey;
        GPUSelectExtKey<T> selectKey;
    };

    struct GPUDReluKey
    {
        GPUDCFKey dcfKey;
        u32 *dReluMask;
    };

    template <typename T>
    struct GPU2RoundReLUKey
    {
        int bin, bout, N;
        GPUDReluKey dreluKey;
        GPUSelectKey<T> selectKey;
    };

    template <typename T>
    struct GPUReluExtendKey
    {
        int bin, bout, N;
        GPUDReluKey dReluKey;
        u32 *dcfMask;
        T *oneHot;
        T *outMask;
    };

    template <typename T>
    GPUSelectExtKey<T> readGPUSelectExtKey(uint8_t** key_as_bytes, int N) {
        GPUSelectExtKey<T> k;
        k.N = N;
        size_t memSz = N * sizeof(T);
        k.rm = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.rmd = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.rmu = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.ud = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.m = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.v = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.w = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.z = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        k.rin = (T *) *key_as_bytes;
        *key_as_bytes += memSz;
        return k;
    };

    template <typename T>
    GPUReluExtKey<T> readReluZeroExtKey(u8 **key_as_bytes)
    {
        GPUReluExtKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.dReluKey = dpf::readGPUDReluKey(key_as_bytes);
        k.selectKey = readGPUSelectExtKey<T>(key_as_bytes, k.N);
        return k;
    }

    GPUDReluKey readGPUDReluKey(u8 **key_as_bytes)
    {
        GPUDReluKey k;
        k.dcfKey = readGPUDCFKey(key_as_bytes);
        k.dReluMask = (u32 *)*key_as_bytes;
        // number of 32-bit integers * sizeof(int)
        *key_as_bytes += ((k.dcfKey.bout * k.dcfKey.M - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
        return k;
    }

    template <typename T>
    GPU2RoundReLUKey<T> readTwoRoundReluKey(u8 **key_as_bytes)
    {
        GPU2RoundReLUKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        size_t memSz = k.N * sizeof(T);
        k.dreluKey = readGPUDReluKey(key_as_bytes);
        k.selectKey = readGPUSelectKey<T>(key_as_bytes, k.N);
        return k;
    }

    template <typename T>
    GPUReluExtendKey<T> readGPUReluExtendKey(u8 **key_as_bytes)
    {
        GPUReluExtendKey<T> k;
        memcpy(&k, *key_as_bytes, 3 * sizeof(int));
        *key_as_bytes += (3 * sizeof(int));
        k.dReluKey = readGPUDReluKey(key_as_bytes);
        k.dcfMask = (u32 *)*key_as_bytes;
        int N = k.dReluKey.dcfKey.M;
        *key_as_bytes += ((2 * N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
        k.oneHot = (T *)*key_as_bytes;
        *key_as_bytes += 4 * N * sizeof(T);
        k.outMask = (T *)*key_as_bytes;
        *key_as_bytes += 2 * N * sizeof(T);
        return k;
    }
}

#include "gpu_relu.cu"