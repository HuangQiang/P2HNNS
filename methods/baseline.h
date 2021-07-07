#pragma once

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"
#include "eh.h"
#include "bh.h"
#include "mh.h"
#include <cstddef>

namespace p2h {

// -----------------------------------------------------------------------------
//  Random_Scan: random selection and scan
// -----------------------------------------------------------------------------
class Random_Scan {
public:
    Random_Scan(                    // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        const float *data);             // input data

    // -------------------------------------------------------------------------
    ~Random_Scan();                 // destrctor

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
        ret += sizeof(int)*n_pts_;  // random_id
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   *random_id_;              // randon data index
    const float *data_;             // data objects
};

// -----------------------------------------------------------------------------
//  Sorted_Scan: sort data objects by l2-norms in ascending order and scan
// -----------------------------------------------------------------------------
class Sorted_Scan {
public:
    Sorted_Scan(                    // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        const float *data);             // input data

    // -------------------------------------------------------------------------
    ~Sorted_Scan();                 // destrctor

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
        ret += sizeof(int)*n_pts_;    // sorted_id
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   *sorted_id_;              // sorted data index
    const float *data_;             // data objects
};

// -----------------------------------------------------------------------------
//  Angular Hash: basic data structure for EH, BH, and MH by splitting data 
//  into disjoint partition their l2-norm and normalization
// -----------------------------------------------------------------------------
class Angular_Hash {
public:
    Angular_Hash(                   // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   M,                        // #proj vecotr used for a single hasher
        int   m,                        // #single hasher of the compond hasher
        int   l,                        // #hash tables
        float b,                        // interval ratio
        const float *data);             // input data

    // -------------------------------------------------------------------------
    ~Angular_Hash();                // destrctor

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
        ret += sizeof(int)*n_pts_;  // sorted_id_
        ret += sizeof(int)*block_num_.capacity(); // block_num_

        // index of block_
        for (size_t i = 0; i < block_num_.size(); ++i) {
            ret += sizeof(*hash_[i]);
            ret += hash_[i]->get_memory_usage();
        }
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   M_;                       // #proj vecotr used for a single hasher
    int   m_;                       // #single hasher of the compond hasher
    int   l_;                       // #hash tables
    float b_;                       // interval ratio
    const float *data_;             // data objects

    int *sorted_id_;                // sorted data index
    std::vector<int> block_num_;    // block numbers
    std::vector<Basic_Hash*> hash_; // index of blocks
};

} // end namespace p2h
