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
//  EH_Hash: Embedding Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class EH_Hash : public Basic_Hash<DType> {
public:
    EH_Hash(                        // constructor
        int   n,                        // number of input data
        int   d,                        // dimension of input data
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const int   *index,             // index of input data 
        const float *data);             // input data

    // -------------------------------------------------------------------------
    virtual ~EH_Hash();             // destructor

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
        ret += sizeof(float)*(uint64_t)m_*l_*dim_*dim_; // projv_
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    const int *index_;              // index of input data
    
    float *projv_;                  // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        const float *data,              // input data
        const float *proj) const;       // random projection vector
};

// -----------------------------------------------------------------------------
template<class DType>
EH_Hash<DType>::EH_Hash(            // constructor
    int   n,                            // number of input data
    int   d,                            // dimension of input data
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const int   *index,                 // index of input data
    const float *data)                  // input data
    : n_(n), dim_(d), m_(m), l_(l), index_(index), buckets_(n, l)
{
    // generate random projection vectors
    uint64_t size = (uint64_t) m * l * d * d;
    projv_ = new float[size];
    for (size_t i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
    }
    // build hash tables for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[(uint64_t) i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void EH_Hash<DType>::get_sig_data(  // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*dim_*dim_];
            float val = calc_hash_value(data, proj);

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig << 1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
float EH_Hash<DType>::calc_hash_value(// calc hash value
    const float *data,                  // input data
    const float *proj) const            // random projection vector
{
    float val = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        for (int j = 0; j < dim_; ++j) {
            val += data[i] * data[j] * proj[i*dim_+j];
        }
    }
    return val;
}

// -----------------------------------------------------------------------------
template<class DType>
EH_Hash<DType>::~EH_Hash()          // destructor
{
    delete[] projv_;
}

// -----------------------------------------------------------------------------
template<class DType>
int EH_Hash<DType>::nns(            // point-to-hyperplane NNS
    int   cand,                         // #candidates
    const DType *data,                  // input data
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int cand_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        const DType *point = &data[(uint64_t) index_[idx]*dim_];
        float dist = fabs(calc_ip2<DType>(dim_, point, query));

        list->insert(dist, index_[idx] + 1);
        ++cand_cnt;
    });
    return cand_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void EH_Hash<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*dim_*dim_];
            float val = calc_hash_value(query, proj);

            SigType new_sig = !(val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
//  Orig_EH: Original Embedding Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class Orig_EH {
public:
    using SigType = int32_t;

    // -------------------------------------------------------------------------
    Orig_EH(                        // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~Orig_EH();                     // destructor

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
        ret += sizeof(float)*(uint64_t)m_*l_*dim_*dim_; // projv_
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    const DType *data_;             // data objects

    float *projv_;                  // random projection vectors
    KLBucketingFlat<SigType> buckets_; // hash tables

    // -------------------------------------------------------------------------
    void get_sig_data(              // get the signature of data
        const DType *data,              // input data
        std::vector<SigType> &sig) const;// signature (return)

    // -------------------------------------------------------------------------
    template<class HType>
    float calc_hash_value(          // calc hash value
        const HType *data,              // input data
        const float *proj) const;       // random projection vector

    // -------------------------------------------------------------------------
    void get_sig_query(             // get the signature of query
        const float *query,             // input query 
        std::vector<SigType> &sig) const;// signature (return)
};

// -----------------------------------------------------------------------------
template<class DType>
Orig_EH<DType>::Orig_EH(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const DType *data)                  // input data
    : n_(n), dim_(d), m_(m), l_(l), data_(data), buckets_(n, l)
{
    // generate random projection vectors
    uint64_t size = (uint64_t) m * l * d * d;
    projv_ = new float[size];
    for (size_t i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
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
void Orig_EH<DType>::get_sig_data(  // get signature of data
    const DType *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*dim_*dim_];
            float val = calc_hash_value<DType>(data, proj);

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig << 1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
template<class HType>
float Orig_EH<DType>::calc_hash_value(// calc hash value
    const HType *data,                  // input data
    const float *proj) const            // random projection vector
{
    float val = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        for (int j = 0; j < dim_; ++j) {
            val += (float) data[i] * (float) data[j] * proj[i*dim_+j];
        }
    }
    return val;
}

// -----------------------------------------------------------------------------
template<class DType>
Orig_EH<DType>::~Orig_EH()          // destructor
{
    delete[] projv_;
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_EH<DType>::display()      // display parameters
{
    printf("Parameters of Orig_EH:\n");
    printf("n   = %d\n", n_);
    printf("dim = %d\n", dim_);
    printf("m   = %d\n", m_);
    printf("l   = %d\n", l_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Orig_EH<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int cand_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        const DType *point = &data_[(uint64_t) idx*dim_];
        float dist = fabs(calc_ip2<DType>(dim_, point, query));
        list->insert(dist, idx+1);
        ++cand_cnt;
    });
    return cand_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_EH<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*dim_*dim_];
            float val = calc_hash_value<float>(query, proj);

            SigType new_sig = !(val > 0);
            cur_sig = (cur_sig << 1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

} // end namespace p2h
