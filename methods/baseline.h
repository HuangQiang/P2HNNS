#pragma once

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"
#include "eh.h"
#include "bh.h"
#include "mh.h"
#include <cstddef>
#include <cstdint>

namespace p2h {

// -----------------------------------------------------------------------------
//  Random_Scan: random selection and scan
// -----------------------------------------------------------------------------
template<class DType>
class Random_Scan {
public:
    Random_Scan(                    // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~Random_Scan();                 // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   nc,                       // number of total candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(int)*n_; // random_id
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   *random_id_;              // randon data index
    const DType *data_;             // data objects
};

// -----------------------------------------------------------------------------
template<class DType>
Random_Scan<DType>::Random_Scan(    // constructor
    int   n,                            // number of data points
    int   d,                            // dimension of data points
    const DType *data)                  // input data
    : n_(n), dim_(d), data_(data)
{
    // get random index by random shuffle
    random_id_ = new int[n];

    int i = 0;
    std::iota(random_id_, random_id_+n, i++);
    std::random_shuffle(random_id_, random_id_+n);
}

// -----------------------------------------------------------------------------
template<class DType>
Random_Scan<DType>::~Random_Scan()  // destructor
{
    delete[] random_id_;
}

// -----------------------------------------------------------------------------
template<class DType>
void Random_Scan<DType>::display()  // display parameters
{
    printf("Parameters of Random_Scan:\n");
    printf("n   = %d\n", n_);
    printf("dim = %d\n", dim_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Random_Scan<DType>::nns(        // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   nc,                           // number of total candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    int   verif_cnt = std::min(nc+top_k-1, n_), idx = -1;
    float dist = -1.0f;
    for (int i = 0; i < verif_cnt; ++i) {
        idx  = random_id_[i];
        dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
        list->insert(dist, idx+1);
    }
    return verif_cnt;
}

// -----------------------------------------------------------------------------
//  Sorted_Scan: sort data objects by l2-norms in ascending order and scan
// -----------------------------------------------------------------------------
template<class DType>
class Sorted_Scan {
public:
    Sorted_Scan(                    // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~Sorted_Scan();                 // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   nc,                       // number of total candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(int)*n_; // sorted_id
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   *sorted_id_;              // sorted data index
    const DType *data_;             // data objects
};

// -----------------------------------------------------------------------------
template<class DType>
Sorted_Scan<DType>::Sorted_Scan(    // constructor
    int   n,                            // number of data points
    int   d,                            // dimension of data points
    const DType *data)                  // input data
    : n_(n), dim_(d), data_(data)
{
    // init the sorted_id_ with 0,1,2,...
    sorted_id_ = new int[n];
    int i = 0;
    std::iota(sorted_id_, sorted_id_+n, i++);

    // calc the l2_norms of all data points
    float *l2_norms = new float[n];
    for (i = 0; i < n; ++i) {
        const DType *point = &data[(uint64_t) i*d];
        l2_norms[i] = sqrt(calc_ip<DType>(d-1, point, point));
    }
    // sort data points by their l2-norms in ascending order
    std::sort(sorted_id_, sorted_id_ + n, 
        [&](int i,int j){return l2_norms[i]<l2_norms[j];});

    // release space
    delete[] l2_norms;
}

// -----------------------------------------------------------------------------
template<class DType>
Sorted_Scan<DType>::~Sorted_Scan()  // destructor
{
    delete[] sorted_id_;
}

// -----------------------------------------------------------------------------
template<class DType>
void Sorted_Scan<DType>::display()  // display parameters
{
    printf("Parameters of Sorted_Scan:\n");
    printf("n   = %d\n", n_);
    printf("dim = %d\n", dim_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Sorted_Scan<DType>::nns(        // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   nc,                           // number of total candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    int   verif_cnt = std::min(nc+top_k-1, n_), idx = -1;
    float dist = -1.0f;
    for (int i = 0; i < verif_cnt; ++i) {
        idx  = sorted_id_[i];
        dist = fabs(calc_ip2<DType>(dim_, &data_[(uint64_t)idx*dim_], query));
        list->insert(dist, idx+1);
    }
    return verif_cnt;
}

// -----------------------------------------------------------------------------
//  Angular Hash: basic data structure for EH, BH, and MH by splitting data 
//  into disjoint partition their l2-norm and normalization
// -----------------------------------------------------------------------------
template<class DType>
class Angular_Hash {
public:
    Angular_Hash(                   // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   M,                        // #proj vecotr used for a single hasher
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        float b,                        // interval ratio
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~Angular_Hash();                // destructor

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
        ret += sizeof(int)*n_; // sorted_id_
        ret += sizeof(int)*block_num_.capacity(); // block_num_

        // index of block_
        for (size_t i = 0; i < block_num_.size(); ++i) {
            ret += sizeof(*hash_[i]);
            ret += hash_[i]->get_memory_usage();
        }
        return ret;
    }

protected:
    int   n_;                       // number of data objects
    int   dim_;                     // dimension of data objects
    int   M_;                       // #proj vecotr used for a single hasher
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    float b_;                       // interval ratio
    const DType *data_;             // data objects

    int *sorted_id_;                // sorted data index
    std::vector<int> block_num_;    // block numbers
    std::vector<Basic_Hash<DType>*> hash_; // index of blocks

    // -------------------------------------------------------------------------
    void copy_and_norm(             // copy & normalize orig data to sorted data
        float norm,                     // l2 norm of original data
        const DType *data,              // original data
        float *sorted_data);            // sorted data (return)
};

// -----------------------------------------------------------------------------
template<class DType>
Angular_Hash<DType>::Angular_Hash(  // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const DType *data)                  // input data
    : n_(n), dim_(d), M_(M), m_(m), l_(l), b_(b), data_(data)
{
    // sort data objects by their l2-norms in ascending order
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        const DType *point = &data[(uint64_t) i*d];
        arr[i].id_  = i;
        arr[i].key_ = sqrt(calc_ip<DType>(d, point, point));
    }
    qsort(arr, n, sizeof(Result), ResultComp);

    // get the sorted id and sorted normalized data
    sorted_id_ = new int[n];
    float *sorted_data = new float[(uint64_t) n*d];
    for (int i = 0; i < n; ++i) {
        int   idx  = arr[i].id_;
        float norm = arr[i].key_;

        sorted_id_[i] = idx;
        copy_and_norm(norm, &data[(uint64_t)idx*d], &sorted_data[(uint64_t)i*d]);
    }

    // divide datasets into blocks and build hash tables for each block
    int start = 0;
    while (start < n) {
        // partition block
        int   block_index = start, cnt = 0;
        float max_radius  = arr[start].key_ / b;
        while (block_index < n && arr[block_index].key_ < max_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }
        // add a block
        Basic_Hash<DType> *hash = NULL;
        if (M == 1) {
            hash = new EH_Hash<DType>(cnt, d, m, l, 
                (const int*) sorted_id_ + start, 
                (const float*) sorted_data + (uint64_t) start*d);
        } else if (M == 2) {
            hash = new BH_Hash<DType>(cnt, d, m, l, 
                (const int*) sorted_id_ + start, 
                (const float*) sorted_data + (uint64_t) start*d);
        } else {
            hash = new MH_Hash<DType>(cnt, d, M, m, l, 
                (const int*) sorted_id_ + start, 
                (const float*) sorted_data + (uint64_t) start*d);
        }
        hash_.push_back(hash);
        block_num_.push_back(cnt);
        start += cnt;
    }
    assert(start == n_);
    delete[] arr;
    delete[] sorted_data;
}

// -----------------------------------------------------------------------------
template<class DType>
void Angular_Hash<DType>::copy_and_norm(// copy & normalize orig data to sorted data
    float norm,                         // l2 norm of original data
    const DType *data,                  // original data
    float *sorted_data)                 // sorted data (return)
{
    for (int j = 0; j < dim_; ++j) {
        sorted_data[j] = data[j] / norm;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
Angular_Hash<DType>::~Angular_Hash()// destructor
{
    delete[] sorted_id_;
    if (!hash_.empty()) {
        for (int i = 0; i < hash_.size(); ++i) {
            delete hash_[i];
        }
        std::vector<Basic_Hash<DType>*>().swap(hash_);
    }
    std::vector<int>().swap(block_num_);
}

// -------------------------------------------------------------------------
template<class DType>
void Angular_Hash<DType>::display() // display parameters
{
    printf("Parameters of %s:\n", M_>2 ? "MH" : (M_==2 ? "BH" : "EH"));
    printf("n       = %d\n",   n_);
    printf("dim     = %d\n",   dim_);
    printf("M       = %d\n",   M_);
    printf("m       = %d\n",   m_);
    printf("l       = %d\n",   l_);
    printf("b       = %.2f\n", b_);
    printf("#blocks = %d\n",   (int) block_num_.size());
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Angular_Hash<DType>::nns(       // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    cand += top_k - 1;
    int verif_cnt = 0;

    for (int i = 0; i < block_num_.size(); ++i) {
        // check #candidates according to the ratio of block size
        int block_cand = (int) ceil((float) block_num_[i] * cand / n_);
        block_cand = std::min(block_cand, cand - verif_cnt);
        verif_cnt += hash_[i]->nns(block_cand, data_, query, list);

        if (verif_cnt >= cand) break;
    }
    return verif_cnt;
}

} // end namespace p2h
