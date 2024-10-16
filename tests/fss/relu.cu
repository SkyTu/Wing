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

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/dcf/gpu_relu.h"

#include <cassert>
#include <sytorch/tensor.h>

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bw = 64;
    int N = atoi(argv[3]); //8;
    // const u64 p = (1ULL << 16) - 1;
    const u64 p = 0;
    printf("\n p is %ld", p);
    const u64 q = p;
    int party = atoi(argv[1]);
    printf("\n party is %d", party);
    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);
    printf("\n connected!");
    uint8_t *startPtr, *curPtr;
    printf("\n OneGB is %d\n", OneGB);
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);

    initGPURandomness();
    // d_mask_X 是输入的掩码
    auto d_mask_X = randomGEOnGpu<T>(N, bw);
    // h_mask_X 是在CPU上输入的掩码
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    // d_masked_X 是masked的输入
    auto d_masked_X = getMaskedInputOnGpu(N, bw, d_mask_X, &h_X);

    auto d_temp = dcf::gpuGenTwoRoundReluKey(&curPtr, party, bw, bw, N, d_mask_X, &g);
    printf("Key size=%lu\n", curPtr - startPtr);

    auto d_dreluMask = d_temp.first;
    auto d_reluMask = d_temp.second;

    auto h_mask_O = (T *)moveToCPU((u8 *)d_reluMask, N * sizeof(T), NULL);
    auto k1 = dcf::readTwoRoundReluKey<T>(&startPtr);
    T *d_O;
    for (int i = 0; i < 10; i++)
    {
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        auto d_temp = dcf::gpuTwoRoundRelu(peer, party, k1, d_masked_X, &g, (Stats *)NULL);
        auto d_drelu = d_temp.first;
        gpuFree(d_drelu);
        d_O = d_temp.second;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    }

    auto h_O = (T *)moveToCPU((uint8_t *)d_O, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_O);
    destroyGPURandomness();

    printf("bw=%d, bw=%d, N=%d\n", bw, bw, N);
    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_O[i] - h_mask_O[i]);
        cpuMod(unmasked_O, bw);
        auto o = h_X[i] * (1 - (h_X[i] >> (bw - 1)));
        cpuMod(o, bw);
        // auto o = std::min(std::abs((i64) h_X[i]), (i64) max);
        if (i < 10)
            printf("%d: %ld, %ld, %ld, %ld\n", i, h_X[i], o, unmasked_O, h_mask_X[i]);
        assert(o == unmasked_O);
    }

    return 0;
}