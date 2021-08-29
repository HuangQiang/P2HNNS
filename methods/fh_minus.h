#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "rqalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  FH_Minus: FH without Data Dependent Multi-Partitioning
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, with Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. Without Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
template<class DType>
class FH_Minus {
public:
    FH_Minus(                       // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        int   s,                        // scale factor of dimension
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~FH_Minus();                    // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // separation threshold
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += lsh_->get_memory_usage();
        return ret;
    }

protected:
    int    n_pts_;                  // number of data objects
    int    dim_;                    // dimension of data objects
    int    sample_dim_;             // sample dimension
    int    fh_dim_;                 // new data dimension after transformation
    float  M_;                      // max l2-norm of o' after transformation
    const  DType *data_;            // original data objects

    RQALSH *lsh_;                   // RQALSH for fh data with sampling

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const  DType *data,             // input data
        int    &sample_d,               // sample dimension (return)
        float  &norm,                   // norm of fh_data (return)
        Result *sample_data);           // sample data (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // sample dimension (return)
        Result *sample_query);          // sample query (return)
};

// -----------------------------------------------------------------------------
template<class DType>
FH_Minus<DType>::FH_Minus(          // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const DType *data)                  // input data
    : n_pts_(n), dim_(d), sample_dim_(d*s), fh_dim_(d*(d+1)/2+1), data_(data)
{
    assert(sample_dim_ <= fh_dim_-1);
    lsh_ = new RQALSH(n, fh_dim_, m, NULL);
    M_   = MINREAL;

    // init lsh_ and M_ (build hash tables for rqalsh)
    int    sample_d     = -1; // actual sample dimension
    Result *sample_data = new Result[sample_dim_];
    float  *norm        = new float[n];

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(&data[(uint64_t) i*d], sample_d, norm[i], sample_data);
        if (M_ < norm[i]) M_ = norm[i];

        // calc partial hash value
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(sample_d, j, sample_data);
            lsh_->tables_[j*n+i].id_  = i;
            lsh_->tables_[j*n+i].key_ = val;
        }
    }
    // calc the final hash value
    for (int i = 0; i < n; ++i) {
        float tmp = sqrt(M_ - norm[i]);
        for (int j = 0; j < m; ++j) {
            lsh_->tables_[j*n+i].key_ += lsh_->a_[(j+1)*fh_dim_-1] * tmp;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
    }
    delete[] sample_data;
    delete[] norm;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_Minus<DType>::transform_data(// data transformation
    const  DType *data,                 // input data
    int    &sample_d,                   // sample dimension (return)
    float  &norm,                       // norm of fh_data (return)
    Result *sample_data)                // sample data (return)
{
    // 1: calc probability vector
    float *prob = new float[dim_]; // probability vector
    init_prob_vector<DType>(dim_, data, prob);

    // 2: randomly sample the coordinates for sample_data
    sample_d = 0; norm = 0.0f;
    bool *checked = new bool[fh_dim_];
    memset(checked, false, fh_dim_*sizeof(bool));

    // 2.1: first select the largest coordinate
    int sid = dim_-1;

    checked[sid] = true;
    float key = (float) SQR(data[sid]);
    sample_data[sample_d].id_  = sid;
    sample_data[sample_d].key_ = key;
    norm += SQR(key); ++sample_d;
    
    // 2.2: consider the combination of the remaining coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob); // lower  dim
        int idy = coord_sampling(dim_, prob);   // higher dim
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
    delete[] checked;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
FH_Minus<DType>::~FH_Minus()        // destructor
{
    delete lsh_;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_Minus<DType>::display()     // display parameters
{
    printf("Parameters of FH_Minus:\n");
    printf("n          = %d\n", n_pts_);
    printf("dim        = %d\n", dim_);
    printf("sample_dim = %d\n", sample_dim_);
    printf("fh_dim     = %d\n", fh_dim_);
    printf("m          = %d\n", lsh_->m_);
    printf("max_norm   = %f\n", sqrt(M_));
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int FH_Minus<DType>::nns(           // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, sample_d, sample_query);

    // conduct furthest neighbor search by rqalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->fns(l, cand+top_k-1, MINREAL, sample_d,
        (const Result*) sample_query, cand_list);

    // calc true distance for candidates returned by qalsh
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = cand_list[i];
        float dist = fabs(calc_inner_product2<DType>(dim_, 
            &data_[(uint64_t)idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_Minus<DType>::transform_query(// query transformation
    const  float *query,                // input query
    int    &sample_d,                   // sample dimension (return)
    Result *sample_query)               // sample query (return)
{
    // 1: calc probability vector
    float *prob = new float[dim_];
    init_prob_vector<float>(dim_, query, prob);

    // 2: randomly sample the coordinates for sample_query
    sample_d = 0;
    bool *checked = new bool[fh_dim_];
    memset(checked, false, fh_dim_*sizeof(bool));

    int   sid = -1;
    float key, norm_sample_q = 0.0f;
    for (int i = 0; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob); // lower  dim
        int idy = coord_sampling(dim_, prob);   // higher dim
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue;

            // calc the square coordinates
            checked[sid] = true;
            key = query[idx] * query[idx];
            sample_query[sample_d].id_  = sid;
            sample_query[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates
            checked[sid] = true;
            key = 2 * query[idx] * query[idy];
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
//  FH_Minus_wo_S: Furthest Hyperplane Hash without Data Dependent 
//  Multi-Partitioning & Randomized Sampling
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, without Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. Without Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
template<class DType>
class FH_Minus_wo_S {
public:
    FH_Minus_wo_S(                  // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~FH_Minus_wo_S();               // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // separation threshold
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += lsh_->get_memory_usage();
        return ret;
    }

protected:
    int    n_pts_;                  // number of data objects
    int    dim_;                    // dimension of data objects
    int    fh_dim_;                 // new data dimension after transformation
    float  M_;                      // max l2-norm of o' after transformation
    const  DType *data_;            // original data objects
    
    RQALSH *lsh_;                   // RQALSH for fh data

    // -------------------------------------------------------------------------
    float calc_transform_data_norm( // calc l2-norm-sqr of f(o)
        const DType *data);             // input data object o

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation P*f(o)
        float norm,                     // l2-norm-sqr of f(o)
        const DType *data,              // input data
        float *fh_data);                // fh_data (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const float *query,             // input query
        float *fh_query);               // fh_query (return)
    
    // -------------------------------------------------------------------------
    float calc_transform_query_norm(// calc l2-norm of g(q)
        const float *query);            // input query q
};

// -----------------------------------------------------------------------------
template<class DType>
FH_Minus_wo_S<DType>::FH_Minus_wo_S(// constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    const DType *data)                  // input data
    : n_pts_(n), dim_(d), fh_dim_(d*(d+1)/2+1), data_(data)
{
    // init the max l2-norm-sqr (M_) and calc the l2-norm
    float *norms = new float[n];
    M_ = MINREAL;
    for (int i = 0; i < n_pts_; ++i) {
        norms[i] = calc_transform_data_norm(&data[(uint64_t) i*d]);
        if (M_ < norms[i]) M_ = norms[i];
    }

    // build hash tables for rqalsh
    float *fh_data = new float[fh_dim_];
    lsh_ = new RQALSH(n, fh_dim_, m, NULL);

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(norms[i], &data[(uint64_t) i*d], fh_data);

        // calc the hash values
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(fh_dim_, j, fh_data);
            lsh_->tables_[j*n+i].id_  = i;
            lsh_->tables_[j*n+i].key_ = val;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[i*n], n, sizeof(Result), ResultComp);
    }
    delete[] fh_data;
    delete[] norms;
}

// -----------------------------------------------------------------------------
template<class DType>
float FH_Minus_wo_S<DType>::calc_transform_data_norm(// calc l2-norm-sqr of f(o)
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
void FH_Minus_wo_S<DType>::transform_data(// data transformation
    float norm,                         // l2-norm-sqr of f(o)
    const DType *data,                  // input data
    float *fh_data)                     // fh_data (return)
{
    // calc the square coordinates
    for (int i = 0; i < dim_; ++i) {
        fh_data[i] = (float) SQR(data[i]);
    }
    // calc the differential coordinates
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            fh_data[cnt++] = (float) data[i] * data[j];
        }
    }
    assert(cnt == fh_dim_-1);
    // init the last coordinate
    fh_data[cnt] = sqrt(M_ - norm);
}

// -----------------------------------------------------------------------------
template<class DType>
FH_Minus_wo_S<DType>::~FH_Minus_wo_S()// destructor
{
    delete lsh_;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_Minus_wo_S<DType>::display()// display parameters
{
    printf("Parameters of FH_Minus_wo_S:\n");
    printf("n        = %d\n", n_pts_);
    printf("dim      = %d\n", dim_);
    printf("fh_dim   = %d\n", fh_dim_);
    printf("m        = %d\n", lsh_->m_);
    printf("max_norm = %f\n", sqrt(M_));
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int FH_Minus_wo_S<DType>::nns(      // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    float *fh_query = new float[fh_dim_];
    transform_query(query, fh_query);

    // conduct furthest neighbor search by rqalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->fns(l, cand+top_k-1, MINREAL, 
        (const float*) fh_query, cand_list);

    // calc actual distance for candidates returned by qalsh
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = cand_list[i];
        float dist = fabs(calc_inner_product2<DType>(dim_, 
            &data_[(uint64_t) idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    delete[] fh_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_Minus_wo_S<DType>::transform_query(// query transformation
    const float *query,                 // input query
    float *fh_query)                    // fh_query after transform (return)
{
    float norm_q = calc_transform_query_norm(query);
    float lambda = sqrt(M_ / norm_q);

    // calc the square coordinates
    for (int i = 0; i < dim_; ++i) {
        fh_query[i] = query[i] * query[i] * lambda;
    }
    // calc the differential coordinates
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            fh_query[cnt++] = 2 * query[i] * query[j] * lambda;
        }
    }
    assert(cnt == fh_dim_-1);
    // init the last coordinate
    fh_query[cnt] = 0.0f;
}

// -----------------------------------------------------------------------------
template<class DType>
float FH_Minus_wo_S<DType>::calc_transform_query_norm(// calc l2-norm-sqr of g(q)
    const float *query)                 // input query q
{
    float tmp, norm2 = 0.0f, norm4 = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        tmp = SQR(query[i]);
        norm2 += tmp; norm4 += SQR(tmp);
    }
    return 2*norm2*norm2 - norm4;
}

} // end namespace p2h
