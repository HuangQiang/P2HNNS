#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "../lccs_bucket/bucketAlg/lcs_int.h"

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  NH: Nearest Hyperplane Hashing based on LCCS Bucketing Framework
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, with Randomized Sampling
//  2  Use LCCS Bucketing framework (LCCS-LSH) for P2PNNS
// -----------------------------------------------------------------------------
template<class DType>
class NH {
public:
    using LCCS = mylccs::LCCS_SORT_INT;
    using SigType = int32_t;

    // -------------------------------------------------------------------------
    NH(                             // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hasher
        int   s,                        // scale factor of dimension
        float w,                        // bucket width
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~NH();                          // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*(nh_dim_ + 1); // proja_, projb_
        ret += bucketerp_.get_memory_usage();  // bucketerp_
        ret += data_sigs_.get_memory_usage();  // data_sigs_
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   sample_dim_;              // sample dimension
    int   nh_dim_;                  // new data dimension after transformation
    int   m_;                       // #single hasher of the compond hasher
    float w_;                       // bucket width
    const DType *data_;             // data points

    float M_;                       // max l2-norm-sqr
    float *proja_;                  // random projection vectors
    float *projb_;                  // random shifts
    LCCS bucketerp_;                // hash tables
    NDArray<2, int32_t> data_sigs_; // data signatures

    // -------------------------------------------------------------------------
    void get_sig_data_partial(      // get signature of data
        const DType *data,              // input data object o
        float &norm,                    // l2-norm-sqr of f(o)
        float *projs);                  // projection values (return)

    // -------------------------------------------------------------------------
    float transform_data(           // data transformation
        const  DType *data,             // input data
        int    &sample_d,               // number of sample dimension (return)
        Result *sample_data);           // sample data (return)

    // -------------------------------------------------------------------------
    void get_sig_query(             // get the signature of query
        const float *query,             // input query 
        SigType *sig);                  // signature (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // number of sample dimension (return)
        Result *sample_query);          // sample query (return)
};

// -----------------------------------------------------------------------------
template<class DType>
NH<DType>::NH(                      // constructor
    int   n,                            // number of input data
    int   d,                            // dimension of input data
    int   m,                            // #hashers
    int   s,                            // scale factor of dimension
    float w,                            // bucket width
    const DType *data)                  // input data
    : n_(n), dim_(d), sample_dim_(dim_*s), nh_dim_(d*(d+1)/2+1), m_(m), 
    w_(w), data_(data), bucketerp_(m, 1)
{
    // init hash functions: proja_ and projb_
    int proj_size = m * nh_dim_;
    proja_ = new float[proj_size];
    for (int i = 0; i < proj_size; ++i) proja_[i] = gaussian(0.0f, 1.0f);

    projb_ = new float[m];
    for (int i = 0; i < m; ++i) projb_[i] = uniform(0, w_);
    
    // build hash tables for the hash values of data objects
    M_ = MINREAL;
    float *projs = new float[(uint64_t)n*m];
    float *norms = new float[n];
    
    // calc partial hash values
    for (int i = 0; i < n; ++i) {
        get_sig_data_partial(&data[(uint64_t)i*d], norms[i], 
            &projs[(uint64_t)i*m]);
        if (norms[i] > M_) M_ = norms[i];
    }
    // calc the final hash values with the last coordinate
    data_sigs_.resize({n, m});
    SigType **data_sigs_ptr = data_sigs_.to_ptr();
    for (int i = 0; i < n; ++i) {
        float *vals = &projs[(uint64_t)i*m];
        float last_coord = sqrt(M_ - norms[i]);

        for (int j = 0; j < m; ++j) {
            float val = vals[j] + last_coord * proja_[(j+1)*nh_dim_-1];
            data_sigs_ptr[i][j] = SigType((val + projb_[j]) / w_);
        }
    }
    bucketerp_.build(data_sigs_);

    delete[] norms;
    delete[] projs;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH<DType>::get_sig_data_partial(// get signature of data
    const DType *data,                  // input data object o
    float &norm,                        // l2-norm-sqr of f(o)
    float *projs)                       // projection values (return)
{
    // calc sample_data with data transformation
    int sample_d = -1;
    Result *sample_data = new Result[sample_dim_];
    norm = transform_data(data, sample_d, sample_data);

    // calc the signature of sample_data
    int   idx = -1;
    float val = 0.0f;
    for (int i = 0; i < m_; ++i) {
        const float *proja = (const float*) &proja_[i*nh_dim_];
        val = 0.0f;
        for (int j = 0; j < sample_d; ++j) {
            idx = sample_data[j].id_;
            val += proja[idx] * sample_data[j].key_;
        }
        projs[i] = val;
    }
    delete[] sample_data;
}

// -----------------------------------------------------------------------------
template<class DType>
float NH<DType>::transform_data(     // data transformation
    const  DType *data,                 // input data
    int    &sample_d,                   // number of sample dimension (return)
    Result *sample_data)                // sample data (return)
{
    // 1: calc probability vector and the l2-norm-square of data
    float *prob = new float[dim_]; // probability vector
    init_prob_vector<DType>(dim_, data, prob);

    // 2: randomly sample coordinate of data as the coordinate of sample_data
    float norm = 0.0f; sample_d = 0;
    bool *checked = new bool[nh_dim_];
    memset(checked, false, nh_dim_*sizeof(bool));

    // 2.1: first consider the largest coordinate
    int sid = dim_-1;

    checked[sid] = true;
    float key = (float) SQR(data[sid]);
    sample_data[sample_d].id_  = sid;
    sample_data[sample_d].key_ = key;
    norm += SQR(key); ++sample_d;
    
    // 2.2: consider the combination of the left coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob);
        int idy = coord_sampling(dim_, prob);
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue;

            // calc the square coordinates of sample_data
            checked[sid] = true;
            key = (float) SQR(data[idx]);
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            norm += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue; 

            // calc the differential coordinates of sample_data
            checked[sid] = true;
            key = (float) data[idx] * data[idy];
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            norm += SQR(key); ++sample_d;
        }
    }
    delete[] prob;
    delete[] checked;

    return norm;
}

// -----------------------------------------------------------------------------
template<class DType>
NH<DType>::~NH()          // destructor
{
    delete[] proja_;
    delete[] projb_;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH<DType>::display()      // display parameters
{
    printf("Parameters of NH:\n");
    printf("n          = %d\n", n_);
    printf("dim        = %d\n", dim_);
    printf("sample_dim = %d\n", sample_dim_);
    printf("m          = %d\n", m_);
    printf("w          = %f\n", w_);
    printf("max_norm   = %f\n", sqrt(M_));
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int NH<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(m_);
    get_sig_query(query, &sigs[0]);

    // printf("cand=%d\n", cand);
    int   verif_cnt = 0, step = (cand+m_-1)/m_;
    float dist = -1.0f;
    bucketerp_.for_candidates(step, sigs, [&](int idx) {
        // verify the true distance of idx
        dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
        list->insert(dist, idx+1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query
    SigType* sig)                       // signature (return)
{
    // calc sample_query with query transformation
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, sample_d, sample_query);

    // calc the signature of sample_data
    int   idx = -1;
    float val = 0.0f;
    for (int i = 0; i < m_; ++i) {
        const float *proja = (const float*) &proja_[i*nh_dim_];
        val = 0.0f;
        for (int j = 0; j < sample_d; ++j) {
            idx = sample_query[j].id_;
            val += proja[idx] * sample_query[j].key_;
        }
        sig[i] = SigType((val + projb_[i])/w_);
    }
    delete[] sample_query;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH<DType>::transform_query(    // query transformation
    const  float *query,                // input query
    int    &sample_d,                   // number of sample dimension (return)
    Result *sample_query)               // sample query (return)
{
    // 1: calc probability vector
    float *prob = new float[dim_];
    init_prob_vector<float>(dim_, query, prob);

    // 2: randomly sample the coordinates for sample_query
    sample_d = 0;
    bool *checked = new bool[nh_dim_];
    memset(checked, false, nh_dim_*sizeof(bool));

    int   sid = -1;
    float key, norm_sample_q = 0.0f;
    for (int i = 0; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_, prob);
        int idy = coord_sampling(dim_, prob);
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue; 

            // calc the square coordinates
            checked[sid] = true;
            key = -query[idx] * query[idx];
            sample_query[sample_d].id_  = sid;
            sample_query[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates
            checked[sid] = true;
            key = -2 * query[idx] * query[idy];
            sample_query[sample_d].id_  = sid;
            sample_query[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
    }    
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    for (int i = 0; i < sample_d; ++i) sample_query[i].key_ *= lambda;

    delete[] prob;
    delete[] checked;
}

// -----------------------------------------------------------------------------
//  NH_wo_S: Nearest Hyperplane Hashing based on LCCS Bucketing without 
//  Randomized Sampling
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, without Randomized Sampling
//  2  Use LCCS Bucketing framework (LCCS-LSH) for P2PNNS
// -----------------------------------------------------------------------------
template<class DType>
class NH_wo_S {
public:
    using LCCS = mylccs::LCCS_SORT_INT;
    using SigType = int32_t;

    // -------------------------------------------------------------------------
    NH_wo_S(                        // constructor
        int   n,                        // number of input data
        int   d,                        // dimension of input data
        int   m,                        // #hasher
        float w,                        // bucket width
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~NH_wo_S();                     // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*(nh_dim_ + 1); // proja_, projb_
        ret += bucketerp_.get_memory_usage();  // bucketerp_
        ret += data_sigs_.get_memory_usage();  // data_sigs_
        return ret;
    }

protected:
    int   n_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   nh_dim_;                  // new data dimension after transformation
    int   m_;                       // #single hasher of the compond hasher
    float w_;                       // bucket width
    const DType *data_;             // input data objects

    float M_;                       // max l2-norm-sqr
    float *proja_;                  // random projection vectors
    float *projb_;                  // random shifts
    LCCS bucketerp_;                // hash tables
    NDArray<2, int32_t> data_sigs_; // data signatures

    // -------------------------------------------------------------------------
    float calc_transform_data_norm( // calc l2-norm of f(o)
        const DType *data);             // input data object o

    // -------------------------------------------------------------------------
    void get_sig_data(              // get the signature of data
        float norm,                     // l2-norm-sqr of f(o)
        const DType *data,              // input data
        SigType *sig);                  // signature (return)
    
    // -------------------------------------------------------------------------
    void transform_data(            // data transformation P*f(o)
        float norm,                     // l2-norm-sqr of f(o)
        const DType *data,              // input data
        float *nh_data);                // nh_data (return)

    // -------------------------------------------------------------------------
    void get_sig_query(             // get the signature of query
        const float *query,             // input query 
        SigType *sig);                  // signature (return)
    
    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const float *query,             // input query
        float *nh_query);               // nh_query after transform (return)
    
    // -------------------------------------------------------------------------
    float calc_transform_query_norm(// calc l2-norm of g(q)
        const float *query);            // input query q
};

// -----------------------------------------------------------------------------
template<class DType>
NH_wo_S<DType>::NH_wo_S(            // constructor
    int   n,                            // number of input data
    int   d,                            // dimension of input data
    int   m,                            // #hashers
    float w,                            // bucket width
    const DType *data)                  // input data
    : n_(n), dim_(d), nh_dim_(d*(d+1)/2+1), m_(m), w_(w), data_(data), 
    bucketerp_(m, 1)
{
    // init hash functions: proja_ and projb_
    int proj_size = m * nh_dim_;
    proja_ = new float[proj_size];
    for (int i = 0; i < proj_size; ++i) proja_[i] = gaussian(0.0f, 1.0f);

    projb_ = new float[m];
    for(int i = 0; i < m; ++i) projb_[i] = uniform(0, w_);
    
    // calc l2-norms of f(o) and the max l2-norm-sqr
    float *norms = new float[n]; // l2-norm-sqr of f(o)
    M_ = MINREAL;
    for (int i = 0; i < n; ++i) {
        norms[i] = calc_transform_data_norm(&data[(uint64_t)i*dim_]);
        if (M_ < norms[i]) M_ = norms[i];
    }

    // build hash tables for the hash values of data objects
    data_sigs_.resize({n, m});
    SigType **data_sigs_ptr = data_sigs_.to_ptr();
    for (int i = 0; i < n; ++i) {
        get_sig_data(norms[i], &data[(uint64_t) i*d], data_sigs_ptr[i]);
    }
    bucketerp_.build(data_sigs_);
    delete[] norms;
}

// -----------------------------------------------------------------------------
template<class DType>
float NH_wo_S<DType>::calc_transform_data_norm(// calc l2-norm-sqr of f(o)
    const DType *data)                  // input data object o
{
    float tmp, norm2 = 0.0f, norm4 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        tmp = (float) SQR(data[i]);
        norm2 += tmp; norm4 += SQR(tmp);
    }
    return (norm2 * norm2 + norm4) / 2.0f;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_wo_S<DType>::get_sig_data(  // get signature of data
    float norm,                         // l2-norm-sqr of f(o)
    const DType *data,                  // input data
    SigType *sig)                       // signature (return)
{
    // calc nh_data P*f(o)
    float *nh_data = new float[nh_dim_];
    transform_data(norm, data, nh_data);

    // calc the signature of nh_data
    for (int i = 0; i < m_; ++i) {
        const float *proja = (const float*) &proja_[i*nh_dim_];
        float val = calc_ip2<float>(nh_dim_, nh_data, proja);
        sig[i] = SigType((val + projb_[i])/w_);
    }
    delete[] nh_data;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_wo_S<DType>::transform_data(// data transformation P*f(o)
    float norm,                         // l2-norm-sqr of f(o)
    const DType *data,                  // input data
    float *nh_data)                     // nh_data (return)
{
    // calc the square coordinates
    for (int i = 0; i < dim_; ++i) {
        nh_data[i] = (float) SQR(data[i]);
    }
    // calc the differential coordinates
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            nh_data[cnt++] = (float) data[i] * data[j];
        }
    }
    assert(cnt == nh_dim_-1);
    // init the last coordinate
    nh_data[cnt] = sqrt(M_ - norm);
}

// -----------------------------------------------------------------------------
template<class DType>
NH_wo_S<DType>::~NH_wo_S()          // destructor
{
    delete[] proja_;
    delete[] projb_;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_wo_S<DType>::display()      // display parameters
{
    printf("Parameters of NH_wo_S:\n");
    printf("n        = %d\n", n_);
    printf("dim      = %d\n", dim_);
    printf("m        = %d\n", m_);
    printf("w        = %f\n", w_);
    printf("max_norm = %f\n", sqrt(M_));
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int NH_wo_S<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(m_);
    get_sig_query(query, &sigs[0]);

    // printf("cand=%d\n", cand);
    int   verif_cnt = 0, step = (cand+m_-1)/m_;
    float dist = -1.0f;
    bucketerp_.for_candidates(step, sigs, [&](int idx) {
        // verify the true distance of idx
        dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
        list->insert(dist, idx+1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_wo_S<DType>::get_sig_query( // get signature of query
    const float *query,                 // input query q
    SigType *sig)                       // signature (return)
{
    // calc nh_query Q*g(q)
    float *nh_query = new float[nh_dim_];
    transform_query(query, nh_query);

    // calc the signatures of nh_query
    for (int i = 0; i < m_; ++i) {
        float *proja = &proja_[i*nh_dim_];
        float val = calc_ip2<float>(nh_dim_, nh_query, (const float*) proja);
        sig[i] = SigType((val + projb_[i])/w_);
    }
    delete[] nh_query;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_wo_S<DType>::transform_query(// query transformation Q*g(q)
    const float *query,                 // input query q
    float *nh_query)                    // nh_query after transform (return)
{
    float norm_q = calc_transform_query_norm(query);
    float lambda = sqrt(M_ / norm_q);

    // calc the square coordinates
    for (int i = 0; i < dim_; ++i) {
        nh_query[i] = -query[i] * query[i] * lambda;
    }
    // calc the differential coordinates
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            nh_query[cnt++] = -2 * query[i] * query[j] * lambda;
        }
    }
    assert(cnt == nh_dim_-1);
    // init the last coordinate
    nh_query[cnt] = 0.0f;
}

// -----------------------------------------------------------------------------
template<class DType>
float NH_wo_S<DType>::calc_transform_query_norm(// calc l2-norm-sqr of g(q)
    const float *query)                 // input query q
{
    float tmp, norm2 = 0.0f, norm4 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        tmp = SQR(query[i]);
        norm2 += tmp; norm4 += SQR(tmp);
    }
    return 2 * norm2 * norm2 - norm4;
}

} // end namespace p2h
