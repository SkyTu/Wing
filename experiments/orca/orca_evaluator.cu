// 
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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <omp.h>
#include <string>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "../datasets/gpu_data.h"

#include "nn/orca/gpu_layer.h"
#include "nn/orca/gpu_model.h"

#include "cnn.h"
#include "model_accuracy.h"

#include <sytorch/softmax.h>
#include <sytorch/backend/llama_base.h>

#include "cuda_runtime_api.h"

u64 *gpuSoftmax(int batchSz, int numClasses, int party, SigmaPeer *peer, u64 *d_I, u64 *labels, bool secfloat, LlamaBase<u64> *llama)
{
    Tensor4D<u64> inp(batchSz, numClasses, 1, 1);
    Tensor4D<u64> softmaxOp(batchSz, numClasses, 1, 1);

    size_t memSz = batchSz * numClasses * sizeof(u64);
    moveIntoCPUMem((u8 *)inp.data, (u8 *)d_I, memSz, NULL);
    gpuFree(d_I);
    if (secfloat)
    {
        softmax_secfloat(inp, softmaxOp, dcf::orca::global::scale, LlamaConfig::party);
    }
    else
    {
        pirhana_softmax(inp, softmaxOp, dcf::orca::global::scale);
        // softmax<u64,dcf::orca::global::scale>(inp, softmaxOp);
    }
    for (int img = 0; img < batchSz; img++)
    {
        for (int c = 0; c < numClasses; c++)
        {
            softmaxOp(img, c, 0, 0) -= (labels[numClasses * img + c] * (((1LL << dcf::orca::global::scale)) / batchSz));
        }
    }
    reconstruct(inp.d1 * inp.d2, softmaxOp.data, 64);
    d_I = (u64 *)moveToGPU((u8 *)softmaxOp.data, memSz, NULL);
    return d_I;
}

void trainModel(dcf::orca::GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer, u64 *data, u64 *labels, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch)
{
    auto start = std::chrono::high_resolution_clock::now();
    size_t inpMemSz = m->inpSz * sizeof(u64);
    // printf("data=%p, labels=%p\n", data, labels);
    auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
    u64 *d_O;
    for (int i = 0; i < m->layers.size(); i++)
    {
        m->layers[i]->readForwardKey(keyBuf);
        // printf("readForwardKey %d\n", i);
        d_O = m->layers[i]->forward(peer, party, d_I, g);
        // printf("forward %d\n", i);
        if (d_O != d_I)
            gpuFree(d_I);
        d_I = d_O;
    }
    // printf("Forward pass ");
    checkCudaErrors(cudaDeviceSynchronize());
    d_I = gpuSoftmax(m->batchSz, m->classes, party, peer, d_I, labels, secfloat, llama);
    printf("Softmax pass ");
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        m->layers[i]->readBackwardKey(keyBuf, epoch);
        printf("readBackwardKey %d\n", i);
        d_I = m->layers[i]->backward(peer, party, d_I, g, epoch);
        printf("backward %d\n", i);
    }
}

u64 getKeySz(std::string dir, std::string modelName)
{
    std::ifstream kFile(dir + modelName + ".txt");
    u64 keySz;
    kFile >> keySz;
    return keySz;
}

void rmWeights(std::string lossDir, int party, int l, int k)
{
    assert(std::filesystem::remove(lossDir + "weights_mask_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
    assert(std::filesystem::remove(lossDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
}

void evaluatorE2E(std::string modelName, std::string dataset, int party, std::string ip, std::string weightsFile, bool floatWeights, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, bool fake_offline = true)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPUMemPool();
    initGPURandomness();
    initCPURandomness();
    // assert(epochs < 6);

    omp_set_num_threads(2);

    printf("Sync=%d\n", sync);
    printf("Opening fifos\n");
    char one = 1;
    char two = 2;

    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto weightsDir = lossDir + "weights/";
    auto keySzDir = trainingDir + "keysize/";
    std::ofstream lossFile(lossDir + "loss.txt");
    std::ofstream accFile(lossDir + "accuracy.txt");

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    printf("Model created\n");
    m->initWeights(weightsFile, floatWeights);
    printf("Weights initialized\n");

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(false);
    LlamaBase<u64> *llama = nullptr;

    // automatically truncates by scale
    LlamaConfig::party = party + 2;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);
    
    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party);
    dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    printf("Starting training\n");
    
    Dataset d = readDataset(dataset, party);
    int fd = openForReading(keyFile + "_" + to_string(0) + "_" + to_string(0) + "_" + std::to_string(0) + ".dat");
    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {
            // Open the key file for reading
            printf("Iteration=%u\n", l * blocks * blockSz + k * blockSz);
            // uncomment for end to end run
            peer->sync();
            auto startComm = peer->bytesSent() + peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            for (int j = 0; j < blockSz; j++)
            {
                readKey(fd, keySz, curKeyBuf, &keyReadTime);
                peer->sync();
                auto computeStart = std::chrono::high_resolution_clock::now();
                auto labelsIdx = (k * blockSz + j) * batchSz * d.classes;
                int dataIdx = (k * blockSz + j) * d.H * d.W * d.C * batchSz;
                // printf("Training model %d, %d\n", j, l);
                // printf("Data index=%d, Labels index=%d\n", dataIdx, labelsIdx);
                trainModel(m, &curKeyBuf, party, peer, &(d.data[dataIdx]), &(d.labels[labelsIdx]), &g, secfloat, llama, l);
                auto computeEnd = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
                computeTime += elapsed;  
                curKeyBuf = &keyBuf1[0]; 
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            onlineTime += elapsed;
            printf("Online time (ms): %lu\n", elapsed.count());
            auto endComm = peer->bytesSent() + peer->bytesReceived();
            commBytes += (endComm - startComm);
            std::pair<double, double> res;
            m->dumpWeights(weightsDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + "_" + std::to_string(blockSz-1) + ".dat");
            if (dataset == "mnist")
            {
                printf("Getting loss for MNIST\n");
                res = getLossMNIST<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k, blockSz-1, true);
            }
            else
            {
                printf("Getting loss for CIFAR10\n");
                res = getLossCIFAR10<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k);
            }
            auto accuracy = res.first;
            auto loss = res.second;
            printf("Accuracy=%lf, Loss=%lf\n", accuracy, loss);
            lossFile << loss << std::endl;
            accFile << accuracy << std::endl;   
        }
        m->dumpWeights(weightsDir + "masked_weights_reinit_" + std::to_string(party) + "_" + std::to_string(l+1) + "_" + std::to_string(blocks - 1) + "_" + std::to_string(blockSz - 1) + ".dat");
    }
    close(fd);


    LlamaConfig::peer->close();
    int iterations = epochs * blocks * blockSz;
    commBytes += secFloatComm;
    std::ofstream stats(trainingDir + expName + ".txt");
    auto statsString = "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(secFloatComm);

    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTime = (double)computeTime.count() / (double)iterations;

    double commPerIt = (double)commBytes / (double)iterations;
    statsString += "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTime);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt);
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    lossFile.close();
    accFile.close();
    destroyCPURandomness();
    destroyGPURandomness();
}


int main(int argc, char *argv[])
{
    sytorch_init();
    int party = 0;
    auto ip = "10.176.34.171";
    auto keyDir = std::string(argv[1]);
    using T = u64;
    // Neha: need to fix this later 
    int epochs = 1;
    int blocks = 46;
    int blockSz = 10; // 600
    int batchSz = 128;
    evaluatorE2E("CNN2", "mnist", party, ip, "weights/CNN2.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, false, false, keyDir);
    // evaluatorE2E("P-SecureML", "mnist", party, ip, "weights/PSecureMlNoRelu.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, false, false, keyDir);
    return 0;
}
