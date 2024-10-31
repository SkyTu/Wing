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

#include "utils/gpu_data_types.h"

template <typename T>
struct GPUSelectKey
{
    int N;
    T *a, *b, *c, *d1, *d2;
};

template <typename T>
struct GPUSelectExtendKey
{
    int N;
    T *v, *p, *q, *rb, *rin, *rin_msb, *rout;
};


template <typename T>
GPUSelectExtendKey<T> readGPUSelectExtendKey(uint8_t** key_as_bytes, int N) {
    GPUSelectExtendKey<T> k;
    k.N = N;

    size_t size_in_bytes = N * sizeof(T);

    k.rb = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.rin = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.rin_msb = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.rout = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.v = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.p = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.q = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}

template <typename T>
GPUSelectKey<T> readGPUSelectKey(uint8_t** key_as_bytes, int N) {
    GPUSelectKey<T> k;
    k.N = N;

    size_t size_in_bytes = N * sizeof(T);

    k.a = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.c = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d1 = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d2 = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}

#include "gpu_select.cu"