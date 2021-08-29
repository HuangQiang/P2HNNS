#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  BH_Hash: Bilinear Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class BH_Hash : public Basic_Hash<DType> {
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
        const DType *data,              // input data
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    using SigType = int32_t;

    virtual void get_sig_data(      // get the signature of data
        const float *data,              // input data
        std::vector<SigType> &sig) const; // signature (return)

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
template<class DType>
BH_Hash<DType>::BH_Hash(            // constructor
    int   n,                            // number of input data
    int   d,                            // dimension of input dataa
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const int   *index,                 // index of input data 
    const float *data)                  // input data
    : n_pts_(n), dim_(d), m_(m), l_(l), index_(index), buckets_(n, l)
{
    // sample random projection variables
    uint64_t size = (uint64_t) m * l * d;
    projv_.resize(size);
    proju_.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
        proju_[i] = gaussian(0.0f, 1.0f);
    }
    
    // build hash table for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[(uint64_t) i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void BH_Hash<DType>::get_sig_data(  // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    uint64_t pidx = 0;
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection  = 0.0f;
            float projection2 = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                pidx = ((uint64_t) ll * m_ + mm) * dim_ + i;
                projection  += data[i] * projv_[pidx];
                projection2 += data[i] * proju_[pidx];
            }
            SigType new_sig = (projection * projection2 > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
BH_Hash<DType>::~BH_Hash()          // destructor
{
}

// -------------------------------------------------------------------------
template<class DType>
int BH_Hash<DType>::nns(            // point-to-hyperplane NNS
    int   cand,                         // #candidates
    const DType *data,                  // input data
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int   verif_cnt = 0, did = -1;
    float dist = -1.0f;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        did  = index_[idx];
        dist = fabs(calc_inner_product2<DType>(dim_, &data[(uint64_t)did*dim_],
            query));
        list->insert(dist, did + 1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void BH_Hash<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    uint64_t pidx = 0;
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection  = 0.0f;
            float projection2 = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                pidx = ((uint64_t) ll * m_ + mm) * dim_ + i;
                projection  += query[i] * projv_[pidx];
                projection2 += query[i] * proju_[pidx];
            }
            SigType new_sig = !(projection * projection2 > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
//  Orig_BH: Original Bilinear Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class Orig_BH {
public:
    using SigType = int32_t;
    
    // -------------------------------------------------------------------------
    Orig_BH(                        // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const DType *data);             // data objects

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
    const DType *data_;             // data objects

    std::vector<float> projv_;      // random projection vectors
    std::vector<float> proju_;      // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables

    // -------------------------------------------------------------------------
    void get_sig_data(              // get the signature of data
        const DType *data,              // input data
        std::vector<SigType> &sig) const;// signature (return)

    // -------------------------------------------------------------------------
    void get_sig_query(             // get the signature of query
        const float *query,             // input query 
        std::vector<SigType> &sig) const;// signature (return)
};

// -----------------------------------------------------------------------------
template<class DType>
Orig_BH<DType>::Orig_BH(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const DType *data)                  // input data
    : n_pts_(n), dim_(d), m_(m), l_(l), data_(data), buckets_(n, l)
{
    // sample random projection variables
    int size = m * l * d;
    projv_.resize(size);
    proju_.resize(size);
    for (int i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
        proju_[i] = gaussian(0.0f, 1.0f);
    }

    // build hash table for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
Orig_BH<DType>::~Orig_BH()          // destructor
{
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_BH<DType>::get_sig_data(  // get signature of data
    const DType *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection  = 0.0f;
            float projection2 = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                int pidx = (ll * m_ + mm) * dim_ + i;
                projection  += data[i] * projv_[pidx];
                projection2 += data[i] * proju_[pidx];
            }
            SigType new_sig = (projection * projection2 > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_BH<DType>::display()      // display parameters
{
    printf("Parameters of Orig_BH:\n");
    printf("n   = %d\n", n_pts_);
    printf("dim = %d\n", dim_);
    printf("m   = %d\n", m_);
    printf("l   = %d\n", l_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Orig_BH<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int verif_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        float dist = fabs(calc_inner_product2<DType>(dim_, 
            &data_[(uint64_t) idx*dim_], query));
        list->insert(dist, idx + 1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_BH<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection  = 0.0f;
            float projection2 = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                int pidx = (ll * m_ + mm) * dim_ + i;
                projection  += query[i] * projv_[pidx];
                projection2 += query[i] * proju_[pidx];
            }
            SigType new_sig = !(projection * projection2 > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

} // end namespace p2h
