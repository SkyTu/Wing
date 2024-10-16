#pragma once

#include "gpu_comms.h"
#include "rss3_struct.h"
#include <sytorch/tensor.h>
#include <llama/comms.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "gpu_stats.h"

extern size_t OneGB;
// extern u8 *h_bufA0, *h_bufA1;
// extern size_t commBufSize;
class Rss3Cluster
{
public:
    size_t commBufSize = 5 * OneGB;
    u8 *h_bufA0, *h_bufA1;
    u8 *sendBuf = nullptr;
    size_t sendSz;
    GpuPeer *peer0;
    GpuPeer *peer1;
    std::string ip0;
    std::string ip1;
    std::string ip2;
    int port0;
    int port1;
    int port2;
    int party;
    bool compress;
    std::mutex mtx;
    std::condition_variable cv;
    bool terminate = false;
    std::atomic<bool> sendHasWork;
    std::thread sendThread;

    Rss3Cluster(int party, bool pinMem, bool compress, std::string ip0, int port0, std::string ip1, int port1, std::string ip2, int port2){
        this->party = party;
        this->ip0 = ip0;
        this->ip1 = ip1;
        this->ip2 = ip2;
        this->port0 = port0;
        this->port1 = port1;
        this->port2 = port2;
        this->peer0 = new GpuPeer(true);
        this->peer1 = new GpuPeer(true);
        initCommBufs(pinMem);
        this->sendHasWork = false;
        connect();
    };

    void connect(){
        if (this->party == 0) {
            this->peer0->connect(0, this->ip1, this->port1);
            this->peer1->connect(0, this->ip2, this->port2);
        } else if (this->party == 1) {
            this->peer0->connect(0, this->ip0, this->port0);
            this->peer1->connect(0, this->ip2, this->port2);
        } else if (this->party == 2) {
            this->peer0->connect(0, this->ip0, this->port0);
            this->peer1->connect(0, this->ip1, this->port1);
        }
    };

    void initCommBufs(bool pinMem){
            // printf("################## Increase the size of comm bufs! #####################\n");
        printf("Allocating %lu bytes of memory for comm bufs\n", commBufSize);
        h_bufA0 = cpuMalloc(commBufSize, pinMem);
        h_bufA1 = cpuMalloc(commBufSize, pinMem);
    };
    
    void freeCommBufs(bool pinMem){
        cpuFree(h_bufA0, pinMem);
        cpuFree(h_bufA1, pinMem);
    };

    template <typename T>
    void convertF2R(T *share_0, T *share_1, RSSShare<T> rss_share ,int bw, u64 N, bool is_share, Stats *s)
    {   
        // Server0: share_0: fss_share, share_1: 2^n - mask_share_0
        // Server1: share_0: fss_share, share_1: 2^n - mask_share_1
        // Dealer: share_0: 2^n - mask_share_0, share_1: 2^n - mask_share_1
    
        // 如果是秘密分享，先进行reconstruct
        if(is_share){
            if(party == SERVER0 || party == SERVER1){
                // 对于SERVER0和SERVER1，share_0传入的是fss_share，share_1传入的是mask_share
                // 如果是share的形式，那么就需要reconstruct
                this->peer0->reconstructInPlace(share_0, bw, N, s);
            }
        }
        // 将rss_share转换为fss_share
        if(party == SERVER0){
            // 对于SERVER0，rss_share.share0是mask_share(即参数里的share1)，rss_share.share1是masked revealed value
            rss_share.share0 = share_1; // mask_share_0
            rss_share.share1 = share_0;
        }
        else if(party == SERVER1){
            rss_share.share0 = share_0;
            rss_share.share1 = share_1; // mask_share_1
        }
        else if(party == DEALER){
            rss_share.share0 = share_1; // mask_share_1
            rss_share.share1 = share_0; // mask_share_0
        }
        else{
            std::cout << "Error: party is not valid!" << std::endl;
        }
        return;
    };

    template <typename T>
    void convertR2F(T *share, T *outMask, RSSShare<T> rss_share ,int bw, u64 N, bool is_share, Stats *s)
    {   
        // 将rss_share转换为revealed masked value(Naive)
        assert(party == SERVER0 || party == SERVER1);
        for(int i = 0; i < N; i++){
            share[i] = rss_share.share1[i] + rss_share.share0[i] + outMask[i];
            share[i] = cpuMod(share[i], bw);
        }
        this->peer0->reconstructInPlace(share, bw, N, s);
    };

    inline void close()
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            terminate = true;
        }
        cv.notify_one();
        sendThread.join();
        peer0->close();
        peer1->close();
    };

};

