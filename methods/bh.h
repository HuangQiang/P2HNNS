#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  BH_Hash: Bilinear Hyperplane Hash
// -----------------------------------------------------------------------------
class BH_Hash : public Basic_Hash {
public:
    BH_Hash(                        // constructor
        int   n,                        // number of input data
        int   d,                        // dimension of input data
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const int   *index,             // index of input data 
        const float *data);             // input data

    // -------------------------------------------------------------------------
    virtual ~BH_Hash();             // destructor

    // -------------------------------------------------------------------------
    virtual int nns(                // point-to-hyperplane NNS
        int   cand,                     // #candidates
        const float *data,              // input data
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    virtual void get_sig_data(      // get the signature of data
        const float *data,              // input data
        std::vector<SigType> &sig) const; // signature (return)

    // -------------------------------------------------------------------------
    virtual void get_sig_query(     // get the signature of query
        const float *query,             // input query 
        std::vector<SigType> &sig) const; // signature (return)

    // -------------------------------------------------------------------------
    virtual uint64_t get_memory_usage() { // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += buckets_.get_memory_usage(); // buckets_
        ret += sizeof(float)*projv_.capacity() * 2; // projv_, proju_
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    const int *index_;              // index of input data

    std::vector<float> projv_;      // random projection vectors
    std::vector<float> proju_;      // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables
};

// -----------------------------------------------------------------------------
//  Orig_BH: Original Bilinear Hyperplane Hash
// -----------------------------------------------------------------------------
class Orig_BH {
public:
    Orig_BH(                        // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const float *data);             // data objects

    // -------------------------------------------------------------------------
    ~Orig_BH();                     // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    using SigType = int32_t;

    void get_sig_data(              // get the signature of data
        const float *data,              // input data
        std::vector<SigType> &sig) const;// signature (return)

    void get_sig_query(             // get the signature of query
        const float *query,             // input query 
        std::vector<SigType> &sig) const;// signature (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += buckets_.get_memory_usage(); // buckets_
        ret += sizeof(float)*projv_.capacity() * 2; // projv_, proju_
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    const float *data_;             // data objects

    std::vector<float> projv_;      // random projection vectors
    std::vector<float> proju_;      // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables
};

} // end namespace p2h
