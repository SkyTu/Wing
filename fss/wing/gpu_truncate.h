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
#include "gpu_relu.h"
#include "../gpu_relu.h"
#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

namespace wing
{
    enum TruncateType
    {
        None,
        // LocalLRS,
        RevealedStochasticTR,
        StochasticTR,
        LocalARS,
        RevealedStochasticTruncate,
        StochasticTruncate,
    };

    using GPUMaskedDCFKey = wing::GPUDReluKey;
    using GPUMaskedDPFKey = dpf::GPUDReluKey;
    const auto readGPUMaskedDCFKey = wing::readGPUDReluKey;
    const auto readGPUMaskedDPFKey = dpf::readGPUDReluKey;
    template <typename T>
    struct GPUTReKey
    {
        int bin, bout, shift, N;
    };


    template <typename T>
    struct GPUZeroExtKey
    {
        int bin, bout, N;
        T *u, *m;
    };

    template <typename T>
    struct GPUTruncateKey
    {
        GPUTReKey<T> TReKey;
        GPUZeroExtKey<T> ZeroExtKey;
    };


    template <typename T>
    GPUZeroExtKey<T> readGPUZeroExtKey(uint8_t **key_as_bytes)
    {
        GPUZeroExtKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);
        size_t memSz = k.N * sizeof(T);
        k.u = (T *)*key_as_bytes;
        *key_as_bytes += memSz;
        k.m = (T *)*key_as_bytes;
        *key_as_bytes += memSz;
        return k;
    }


    // New added function for RFSS3
    template <typename T>
    GPUTReKey<T> readGPUStTRKey(u8 **key_as_bytes)
    {
        GPUTReKey<T> k;
        memcpy(&k, *key_as_bytes, 4 * sizeof(int));
        *key_as_bytes += 4 * sizeof(int);
        return k;
    }
    
    template <typename T>
    GPUTruncateKey<T> readGPUTrStochasticKey(u8 **key_as_bytes)
    {
        GPUTruncateKey<T> k;
        k.TReKey = readGPUStTRKey<T>(key_as_bytes);
        k.ZeroExtKey = readGPUZeroExtKey<T>(key_as_bytes);
        return k;
    }

    template <typename T>
    GPUTruncateKey<T> readGPUTruncateKey(TruncateType t, uint8_t **key_as_bytes)
    {
        GPUTruncateKey<T> k;
        switch (t)
        {
        case TruncateType::RevealedStochasticTruncate:
            k = readGPUTrStochasticKey<T>(key_as_bytes);
            break;
        case TruncateType::StochasticTruncate:
            k = readGPUTrStochasticKey<T>(key_as_bytes);
            break;
        case TruncateType::StochasticTR:
            k.TReKey = readGPUStTRKey<T>(key_as_bytes);
            break;
        case TruncateType::RevealedStochasticTR:
            k.TReKey = readGPUStTRKey<T>(key_as_bytes);
            break;
        default:
            assert(t == TruncateType::None || t == TruncateType::LocalARS || t == TruncateType::StochasticTR || t == TruncateType::RevealedStochasticTR);
        }
        return k;
    }
}
#include "gpu_truncate.cu"