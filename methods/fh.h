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
//  FH: Furthest Hyperplane Hashing
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, with Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. With Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
template<class DType>
class FH {
public:
    FH(                             // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        int   s,                        // scale factor of dimension
        float b,                        // interval ratio
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~FH();                          // destructor

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
        ret += sizeof(float)*fh_dim_; // centroid_
        ret += sizeof(int)*n_;        // shift_id_
        for (auto hash : hash_) {     // blocks_
            ret += hash->get_memory_usage();
        }
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   sample_dim_;              // sample dimension
    int   fh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm sqr of o'
    const DType *data_;             // original data objects

    int *shift_id_;                 // shift data id
    std::vector<RQALSH*> hash_;     // blocks

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const  DType *data,             // input data
        int    &sample_d,               // sample dimension (return)
        float  &norm,                   // l2-norm-sqr of sample_data (return)
        Result *sample_data,            // sample data (return)
        float  *centroid);              // centroid (return)

    // -------------------------------------------------------------------------
    float calc_transform_dist(      // calc l2-dist after transformation
        int   sample_d,                 // dimension of sample data
        float last,                     // the last coordinate of sample data
        float norm_sqr_ctrd,            // the l2-norm-sqr of centroid
        const Result *sample_data,      // sample data
        const float *centroid);         // centroid after data transformation

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // dimension of sample query (return)
        Result *sample_query);          // sample query after transform (return)
};

// -----------------------------------------------------------------------------
template<class DType>
FH<DType>::FH(                      // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // number of hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const DType *data)                  // input data
    : n_(n), dim_(d), sample_dim_(d*s), fh_dim_(d*(d+1)/2+1), data_(data)
{
    // -------------------------------------------------------------------------
    //  calc centroid, l2-norm, and max l2-norm
    // -------------------------------------------------------------------------
    float  *norm        = new float[n];       // l2-norm of sample_data
    float  *centroid    = new float[fh_dim_]; // centroid of sample_data
    int    *sample_d    = new int[n];         // number of sample dimensions
    Result *sample_data = new Result[(uint64_t) n*sample_dim_]; // sample data

    // calc l2-norm & update max l2-norm-sqr
    memset(centroid, 0.0f, fh_dim_*sizeof(float));
    M_ = MINREAL;
    for (int i = 0; i < n; ++i) {
        transform_data(&data[(uint64_t)i*d], sample_d[i], norm[i],
            &sample_data[(uint64_t)i*sample_dim_], centroid);
        if (M_ < norm[i]) M_ = norm[i];
    }

    // calc centroid and its l2-norm-sqr
    float norm_sqr_ctrd = 0.0f;
    for (int i = 0; i < fh_dim_-1; ++i) {
        centroid[i] /= n;
        norm_sqr_ctrd += SQR(centroid[i]);
    }
    float last = 0.0f;
    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(M_ - norm[i]);
        last += norm[i];
    }
    last /= n;
    centroid[fh_dim_-1] = last;
    norm_sqr_ctrd += SQR(last);

    // -------------------------------------------------------------------------
    //  determine shift_id after shifting data objects to centroid
    // -------------------------------------------------------------------------
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        arr[i].id_  = i;
        arr[i].key_ = calc_transform_dist(sample_d[i], norm[i], norm_sqr_ctrd, 
            &sample_data[(uint64_t) i*sample_dim_], centroid);
    }
    qsort(arr, n, sizeof(Result), ResultCompDesc);

    shift_id_ = new int[n];
    for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;

    // -------------------------------------------------------------------------
    //  divide datasets into blocks and build hash tables for each block
    // -------------------------------------------------------------------------
    int start = 0;
    while (start < n) {
        // partition block
        float min_radius  = b * arr[start].key_;
        int   block_index = start, cnt = 0;
        while (block_index < n && arr[block_index].key_ > min_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }

        // add block
        const int *index = (const int*) shift_id_ + start;
        RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);
        for (int i = 0; i < cnt; ++i) {
            // calc the hash values of P*f(o)
            int idx = index[i];
            for (int j = 0; j < m; ++j) {
                float val = hash->calc_hash_value( sample_d[idx], j, norm[idx], 
                    &sample_data[(uint64_t)idx*sample_dim_]);
                hash->tables_[(uint64_t)j*cnt+i].id_  = i;
                hash->tables_[(uint64_t)j*cnt+i].key_ = val;
            }
        }
        // sort hash tables in ascending order of hash values
        for (int i = 0; i < m; ++i) {
            qsort(&hash->tables_[(uint64_t)i*cnt], cnt, sizeof(Result), ResultComp);
        }
        hash_.push_back(hash);
        start += cnt;
    }
    assert(start == n);
    delete[] arr;
    delete[] sample_data;
    delete[] sample_d;
    delete[] centroid;
    delete[] norm;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::transform_data(     // data transformation
    const  DType *data,                 // input data
    int    &sample_d,                   // sample dimension (return)
    float  &norm,                       // l2-norm-sqr of sample_data (return)
    Result *sample_data,                // sample data (return)
    float  *centroid)                   // centroid (return)
{
    // 1: calc probability vector and the l2-norm-square of data
    float *prob = new float[dim_]; // probability vector
    init_prob_vector<DType>(dim_, data, prob);

    // 2: randomly sample the coordinates for sample_data
    sample_d = 0; norm = 0.0f;
    bool *checked = new bool[fh_dim_];
    memset(checked, false, fh_dim_*sizeof(bool));
    
    // 2.1: first select the largest coordinate
    int sid = dim_-1; // sample id

    checked[sid] = true;
    float key = (float) SQR(data[sid]);
    sample_data[sample_d].id_  = sid;
    sample_data[sample_d].key_ = key;
    centroid[sid] += key; norm += SQR(key); ++sample_d;

    // 2.2: consider the combination of the remain coordinates
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
            centroid[sid] += key; norm += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates of sample_data
            checked[sid] = true;
            key = (float) data[idx] * data[idy];
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            centroid[sid] += key; norm += SQR(key); ++sample_d;
        }
    }
    delete[] checked;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
float FH<DType>::calc_transform_dist(// calc l2-dist after transform
    int   sample_d,                     // dimension of sample data
    float last,                         // the last coordinate of sample data
    float norm_sqr_ctrd,                // the l2-norm-sqr of centroid
    const Result *sample_data,          // sample data
    const float *centroid)              // centroid after data transformation
{
    int   idx  = -1;
    float dist = norm_sqr_ctrd, tmp, diff;
    
    // calc the distance for the sample dimension
    for (int i = 0; i < sample_d; ++i) {
        idx  = sample_data[i].id_; tmp = centroid[idx];
        diff = sample_data[i].key_ - tmp;

        dist -= SQR(tmp);
        dist += SQR(diff);
    }
    // calc the distance for the last coordinate
    tmp  = centroid[fh_dim_-1];
    dist -= SQR(tmp);
    dist += SQR(last - tmp);

    return sqrt(dist);
}

// -----------------------------------------------------------------------------
template<class DType>
FH<DType>::~FH()                    // destructor
{
    delete[] shift_id_;
    if (!hash_.empty()) {
        for (auto& hash : hash_) delete hash;
        std::vector<RQALSH*>().swap(hash_);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::display()           // display parameters
{
    printf("Parameters of FH:\n");
    printf("n          = %d\n", n_);
    printf("dim        = %d\n", dim_);
    printf("sample_dim = %d\n", sample_dim_);
    printf("fh_dim     = %d\n", fh_dim_);
    printf("max_norm   = %f\n", sqrt(M_));
    printf("#blocks    = %d\n", (int) hash_.size());
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int FH<DType>::nns(                 // point-to-hyperplane NNS
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
    
    // point-to-hyperplane NNS
    int   idx, size, verif_cnt = 0, n_cand = cand+top_k-1;
    float kfn_dist, kdist, dist, fix_val = 2*M_;
    std::vector<int> cand_list;
    
    for (auto hash : hash_) {
        // check candidates returned by rqalsh
        kfn_dist = -1.0f;
        if (list->isFull()) {
            kdist = list->max_key();
            kfn_dist = sqrt(fix_val - 2*kdist*kdist);
        }
        size = hash->fns(l, n_cand, kfn_dist, sample_d, sample_query, cand_list);
        for (int j = 0; j < size; ++j) {
            idx  = cand_list[j];
            dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
            list->insert(dist, idx + 1);
        }
        // update info
        verif_cnt += size; n_cand -= size;
        if (n_cand <= 0) break;
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::transform_query(    // query transformation
    const  float *query,                // input query
    int    &sample_d,                   // dimension of sample query (return)
    Result *sample_q)                   // sample query after transform (return)
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
            sample_q[sample_d].id_  = sid;
            sample_q[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates
            checked[sid] = true;
            key = 2 * query[idx] * query[idy];
            sample_q[sample_d].id_  = sid;
            sample_q[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
    }
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    for (int i = 0; i < sample_d; ++i) sample_q[i].key_ *= lambda;

    delete[] prob;
    delete[] checked;
}

// -----------------------------------------------------------------------------
//  FH_wo_S: Furthest Hyperplane Hashing without Randomized Sampling
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, without Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. With Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
template<class DType>
class FH_wo_S {
public:
    FH_wo_S(                        // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        float b,                        // interval ratio
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~FH_wo_S();                     // destructor

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
        ret += sizeof(float)*fh_dim_; // centroid_
        ret += sizeof(int)*n_;        // shift_id_
        for (auto hash : hash_) {     // blocks_
            ret += hash->get_memory_usage();
        }
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   fh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm sqr of o'
    const DType *data_;             // original data objects

    int *shift_id_;                 // shift data id
    std::vector<RQALSH*> hash_;     // blocks

    // -------------------------------------------------------------------------
    void calc_transform_centroid(   // calc centorid after data transformation
        const DType *data,              // input data
        float &norm,                    // norm of fh_data (return)
        float *centroid);               // centroid (return)

    // -------------------------------------------------------------------------
    float calc_transform_data_norm( // calc l2-norm-sqr of f(o)
        const DType *data);             // input data object o

    // -------------------------------------------------------------------------
    float calc_transform_dist(      // calc l2-dist after data transformation 
        float last,                     // the last coordinate of P*f(o)
        const DType *data,              // input data object o
        const float *centroid);         // centroid after data transformation

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        float last,                     // the last coordinate of P*f(o)
        const DType *data,              // input data
        float *fh_data);                // fh data (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const float *query,             // input query
        float *fh_query);               // fh_query after transform (return)

    // -------------------------------------------------------------------------
    float calc_transform_query_norm(// calc l2-norm-sqr of g(q)
        const float *query);            // input query q
};

// -----------------------------------------------------------------------------
template<class DType>
FH_wo_S<DType>::FH_wo_S(            // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // number of hash tables
    float b,                            // interval ratio
    const DType *data)                  // input data
    : n_(n), dim_(d), fh_dim_(d*(d+1)/2+1), data_(data)
{
    int   fh_dim_1  = fh_dim_ - 1;
    float *norm     = new float[n];        // l2-norm  of f(o)
    float *centroid = new float[fh_dim_];  // centroid of P*f(o)

    // calc centroid, l2-norm, and max l2-norm
    memset(centroid, 0.0f, fh_dim_*sizeof(float));
    M_ = MINREAL;
    for (int i = 0; i < n; ++i) {
        calc_transform_centroid(&data[(uint64_t) i*d], norm[i], centroid);
        if (M_ < norm[i]) M_ = norm[i];
    }
    for (int i = 0; i < fh_dim_1; ++i) centroid[i] /= n;

    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(M_ - norm[i]);
        centroid[fh_dim_1] += norm[i];
    }
    centroid[fh_dim_1] /= n;

    // determine shift_id based on the descending order of dist(data,centroid)
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        arr[i].id_  = i;
        arr[i].key_ = calc_transform_dist(norm[i], &data[(uint64_t) i*d], 
            centroid);
    }
    qsort(arr, n, sizeof(Result), ResultCompDesc);

    shift_id_ = new int[n];
    for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;

    // divide datasets into blocks and build hash tables for each block
    float *fh_data = new float[fh_dim_]; // P*f(o)
    int   start = 0;
    while (start < n) {
        // partition block
        float min_radius  = b * arr[start].key_;
        int   block_index = start, cnt = 0;
        while (block_index < n && arr[block_index].key_ > min_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }

        // add block
        const int *index = (const int*) shift_id_ + start;
        RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);
        for (int i = 0; i < cnt; ++i) {
            // calc hash values
            int idx = index[i];
            transform_data(norm[idx], &data[(uint64_t)idx*d], fh_data);
            for (int j = 0; j < m; ++j) {
                float val = hash->calc_hash_value(fh_dim_, j, fh_data);
                hash->tables_[(uint64_t)j*cnt+i].id_  = i;
                hash->tables_[(uint64_t)j*cnt+i].key_ = val;
            }
        }
        // sort hash tables in ascending order of hash values
        for (int i = 0; i < m; ++i) {
            qsort(&hash->tables_[(uint64_t)i*cnt], cnt, sizeof(Result), ResultComp);
        }
        hash_.push_back(hash);
        start += cnt;
    }
    assert(start == n);
    delete[] fh_data;
    delete[] norm;
    delete[] centroid;
    delete[] arr;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_wo_S<DType>::calc_transform_centroid(// calc centroid of f(o)
    const DType *data,                  // input data
    float &norm,                        // norm of fh_data (return)
    float *centroid)                    // fh data (return)
{
    norm = calc_transform_data_norm(data);

    // accumulate the square coordinates of centroid
    float tmp = -1.0f;
    for (int i = 0; i < dim_; ++i) {
        centroid[i] += (float) SQR(data[i]);
    }
    // accumulate the differential coordinates of centroid
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            centroid[cnt++] += (float) data[i] * data[j];
        }
    }
    assert(cnt == fh_dim_-1);
}

// -----------------------------------------------------------------------------
template<class DType>
float FH_wo_S<DType>::calc_transform_data_norm(// calc l2-norm-sqr of f(o)
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
float FH_wo_S<DType>::calc_transform_dist(// calc l2-dist of P*f(o) and centroid
    float last,                         // the last coordinate of P*f(o)
    const DType *data,                  // input data object o
    const float *centroid)              // centroid after data transformation
{
    // calc the difference in the square coordinates
    float dist = 0.0f, diff = -1.0f;
    for (int i = 0; i < dim_; ++i) {
        diff =  data[i] * data[i] - centroid[i];
        dist += SQR(diff);
    }
    // calc the difference in the differential coordinates
    int cnt = dim_;
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            diff =  data[i] * data[j] - centroid[cnt++];
            dist += SQR(diff);
        }
    }
    assert(cnt == fh_dim_-1);
    // calc the different for the last coordinate
    dist += SQR(last - centroid[cnt]);

    return sqrt(dist);
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_wo_S<DType>::transform_data(// data transformation
    float last,                         // the last coordinate of P*f(o)
    const DType *data,                  // input data
    float *fh_data)                     // fh_data (return)
{
    // calc the square coordinates
    int cnt = 0;
    for (int i = 0; i < dim_; ++i) {
        fh_data[cnt++] = (float) SQR(data[i]);
    }
    // calc the differential coordinates
    for (int i = 0; i < dim_; ++i) {
        for (int j = i + 1; j < dim_; ++j) {
            fh_data[cnt++] = (float) data[i] * data[j];
        }
    }
    assert(cnt == fh_dim_-1);
    // init the last coordinate
    fh_data[cnt] = last;
}

// -----------------------------------------------------------------------------
template<class DType>
FH_wo_S<DType>::~FH_wo_S()          // destructor
{
    delete[] shift_id_;
    if (!hash_.empty()) {
        for (auto hash : hash_) delete hash;
        std::vector<RQALSH*>().swap(hash_);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_wo_S<DType>::display()      // display parameters
{
    printf("Parameters of FH_wo_S:\n");
    printf("n        = %d\n", n_);
    printf("dim      = %d\n", dim_);
    printf("fh_dim   = %d\n", fh_dim_);
    printf("max_norm = %f\n", sqrt(M_));
    printf("#blocks  = %d\n", (int) hash_.size());
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int FH_wo_S<DType>::nns(            // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    float *fh_query = new float[fh_dim_];
    transform_query(query, fh_query);
    
    // point-to-hyperplane NNS
    int   idx, size, verif_cnt = 0, n_cand = cand+top_k-1;
    float kfn_dist, kdist, dist, fix_val = 2*M_;
    std::vector<int> cand_list;
    
    for (auto hash : hash_) {
        // check candidates returned by rqalsh
        kfn_dist = -1.0f;
        if (list->isFull()) {
            kdist = list->max_key();
            kfn_dist = sqrt(fix_val - 2*kdist*kdist);
        }
        size = hash->fns(l, n_cand, kfn_dist, fh_query, cand_list);
        for (int j = 0; j < size; ++j) {
            idx  = cand_list[j];
            dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
            list->insert(dist, idx + 1);
        }
        // update info
        verif_cnt += size; n_cand -= size; 
        if (n_cand <= 0) break;
    }
    delete[] fh_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH_wo_S<DType>::transform_query(// query transformation
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
float FH_wo_S<DType>::calc_transform_query_norm(// calc l2-norm-sqr of g(q)
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
