#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Query-Aware Locality-Sensitive Hashing (QALSH) is used to solve the problem 
//  of c-Approximate Nearest Neighbor (c-ANN) search. 
//  
//  Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 
//  Query-aware locality-sensitive hashing for approximate nearest neighbor 
//  search, Proceedings of the VLDB Endowment (PVLDB), 9(1): 1–12, 2015.
// -----------------------------------------------------------------------------
class QALSH {
public:
    // -------------------------------------------------------------------------
    int    n_pts_;                  // cardinality
    int    dim_;                    // dimensionality
    int    m_;                      // number of hash tables
    float  *a_;                     // hash functions
    Result *tables_;                // hash tables

    // -------------------------------------------------------------------------
    QALSH(                          // constructor
        int n,                          // cardinality
        int d,                          // dimensionality
        int m);                         // number hash tables

    // -------------------------------------------------------------------------
    ~QALSH();                       // destructor

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const float *data);             // one data/query object
    
    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const Result *data);            // sample data

    // -------------------------------------------------------------------------
    int nns(                        // nearest neighbor search
        int   collision_threshold,      // collision threshold
        int   cand,                     // number of candidates
        const float *query,             // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    int nns(                        // nearest neighbor search
        int   collision_threshold,      // collision threshold
        int   cand,                     // number of candidates
        int   sample_dim,               // sample dimension
        const Result *query,            // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {   // get memory usage
        uint64_t ret = 0;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*dim_; // a_
        ret += sizeof(Result)*m_*n_pts_; // tables_
        return ret;
    }
};

} // end namespace p2h
