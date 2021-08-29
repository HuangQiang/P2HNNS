#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  RQALSH: Reverse Query-Aware Locality-Sensitive Hashing for High-Dimensional 
//  Furthest Neighbor Search
//
//  Qiang Huang, Jianlin Feng, Qiong Fang, Wilfred Ng. Two Efficient Hashing 
//  Schemes for High-Dimensional Furthest Neighbor Search. IEEE Transactions 
//  on Knowledge and Data Engineering (TKDE) 29 (12), 2772 - 2785, 2017.
// -----------------------------------------------------------------------------
class RQALSH {
public:
    int   n_;                       // number of input data
    int   dim_;                     // dimension of input data
    int   m_;                       // #hash tables
    const int *index_;              // index of input data
    float *a_;                      // hash functions
    Result *tables_;                // hash tables

    // -------------------------------------------------------------------------
    RQALSH(                         // constructor
        int   n,                        // number of input data
        int   d,                        // dimension of input data
        int   m,                        // #hash tables
        const int *index);              // index of input data

    // -------------------------------------------------------------------------
    ~RQALSH();                      // destructor

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const float *data);             // one data object o'

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const Result *data);            // sample data

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        float last,                     // the last coordinate of input data
        const Result *data);            // sample data

    // -------------------------------------------------------------------------
    int fns(                        // furthest neighbor search
        int   separation_threshold,     // separation threshold
        int   cand,                     // number of candidates
        float R,                        // limited search range
        const float *query,             // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    int fns(                        // furthest neighbor search
        int   separation_threshold,     // separation threshold
        int   cand,                     // number of candidates
        float R,                        // limited search range
        int   sample_dim,               // sample dimension
        const Result *query,            // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    float find_radius(              // find proper radius
        float w,                        // grid width                        
        const int *lpos,                // left  position of query in hash table
        const int *rpos,                // right position of query in hash table
        const float *q_val);            // hash value of query
        
    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*dim_; // a_
        ret += sizeof(Result)*m_*n_;  // tables_
        return ret;
    }
};

} // end namespace p2h
