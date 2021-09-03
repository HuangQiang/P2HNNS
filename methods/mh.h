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
//  MH_Hash: Multilinear Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class MH_Hash : public Basic_Hash<DType> {
public:
    MH_Hash(                        // constructor
        int   n,                        // number of input data 
        int   d,                        // dimension of input data 
        int   M,                        // #proj vecotr used for a single hasher
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const int   *index,             // index of input data 
        const float *data);             // input data

    // -------------------------------------------------------------------------
    virtual ~MH_Hash();             // destructor

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
        ret += buckets_.get_memory_usage();
        ret += sizeof(float)*(uint64_t)m_*l_*M_*dim_; // projv
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   M_;                       // #proj vecotr used for a single hasher
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
MH_Hash<DType>::MH_Hash(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const int   *index,                 // index of input data
    const float *data)                  // input data
    : n_(n), dim_(d), M_(M), m_(m), l_(l), index_(index), buckets_(n, l)
{
    // sample random projection variables
    uint64_t size = (uint64_t) m * l * M * d;
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
void MH_Hash<DType>::get_sig_data(  // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*M_*dim_];
            float val = calc_hash_value(data, proj);

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
float MH_Hash<DType>::calc_hash_value(// calc hash value
    const float *data,                  // input data
    const float *proj) const            // random projection vector
{
    float product = 1.0f;
    for (int i = 0; i < M_; ++i) {
        const float *vec = &proj[i*dim_];
        float val = 0.0f;
        for (int j = 0; j < dim_; ++j) val += data[j] * vec[j];
        
        product *= val;
    }
    return product;
}

// -----------------------------------------------------------------------------
template<class DType>
MH_Hash<DType>::~MH_Hash()          // destructor
{
    delete[] projv_;
}

// -----------------------------------------------------------------------------
template<class DType>
int MH_Hash<DType>::nns(            // point-to-hyperplane NNS
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
        dist = fabs(calc_ip2<DType>(dim_, &data[(uint64_t)did*dim_], query));
        list->insert(dist, did + 1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void MH_Hash<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*M_*dim_];
            float val = calc_hash_value(query, proj);

            SigType new_sig = !(val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}


// -----------------------------------------------------------------------------
//  Orig_MH: Original Multilinear Hyperplane Hash
// -----------------------------------------------------------------------------
template<class DType>
class Orig_MH {
public:
    using SigType = int32_t;

    // -------------------------------------------------------------------------
    Orig_MH(                        // constructor
        int   n,                        // number of input data 
        int   d,                        // dimension of input data 
        int   M,                        // #proj vecotr used for a single hasher
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~Orig_MH();                     // destructor

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
        ret += sizeof(float)*(uint64_t)m_*l_*M_*dim_; // projv
        return ret;
    }

protected:
    int   n_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   M_;                       // #proj vector used for a single hasher
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
Orig_MH<DType>::Orig_MH(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   M,                            // #proj vector used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const DType *data)                  // input data
    : n_(n), dim_(d), M_(M), m_(m), l_(l), data_(data), buckets_(n, l)
{
    // sample random projection variables
    uint64_t size = (uint64_t) m * l * M * d;
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
void Orig_MH<DType>::get_sig_data(  // get signature of data
    const DType *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*M_*dim_];
            float val = calc_hash_value<DType>(data, proj);

            SigType new_sig = (val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
template<class HType>
float Orig_MH<DType>::calc_hash_value(// calc hash value
    const HType *data,                  // input data
    const float *proj) const            // random projection vector
{
    float product = 1.0f;
    for (int i = 0; i < M_; ++i) {
        const float *vec = &proj[i*dim_];
        float val = 0.0f;
        for (int j = 0; j < dim_; ++j) val += data[j] * vec[j];
        
        product *= val;
    }
    return product;
}

// -----------------------------------------------------------------------------
template<class DType>
Orig_MH<DType>::~Orig_MH()          // destructor
{
    delete[] projv_;
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_MH<DType>::display()      // display parameters
{
    printf("Parameters of Orig_MH:\n");
    printf("n   = %d\n", n_);
    printf("dim = %d\n", dim_);
    printf("M   = %d\n", M_);
    printf("m   = %d\n", m_);
    printf("l   = %d\n", l_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Orig_MH<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);

    int   verif_cnt = 0;
    float dist = -1.0f;
    buckets_.for_cand(cand, sigs, [&](int idx){
        // verify the true distance of idx
        dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t) idx*dim_], query));
        list->insert(dist, idx+1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void Orig_MH<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of ret is l
    for (int i = 0; i < l_; ++i) {
        SigType cur_sig = 0;
        for (int j = 0; j < m_; ++j) {
            const float *proj = &projv_[((uint64_t) i*m_+j)*M_*dim_];
            float val = calc_hash_value<float>(query, proj);

            SigType new_sig = !(val > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[i] = cur_sig;
    }
}

} // end namespace p2h
