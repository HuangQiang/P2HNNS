#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "qalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  NH_Counting: Nearest Hyperplane Hashing based on Dynamic Counting Framework
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, with Randomized Sampling
//  2  Use Dynamic Counting Framework (QALSH) for P2PNNS
// -----------------------------------------------------------------------------
class NH_Counting {
public:
    NH_Counting(                    // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        int   s,                        // scale factor of dimension
        const float *data);             // input data

    // -------------------------------------------------------------------------
    ~NH_Counting();                 // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // collision threshold
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
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   scale_;                   // scale factor of dimension
    int   sample_dim_;              // sample dimension
    int   nh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm of o' after transformation
    const float *data_;             // original data objects
    QALSH *lsh_;                    // QALSH for nh_data with sampling

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const  float *data,             // input data
        float  *prob,                   // probability vector
        bool   *checked,                // is checked?
        float  &norm,                   // norm of nh_data (return)
        int    &sample_d,               // sample dimension (return)
        Result *sample_data);           // sample data (return)
    
    // -------------------------------------------------------------------------
    int sampling(                   // sampling coordinate based on prob
        int   d,                        // dimension
        const float *prob);             // input probability

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // sample dimension (return)
        Result *sample_query);          // sample query (return)
};

// -----------------------------------------------------------------------------
//  NH_Counting_wo_S: Nearest Hyperplane Hashing based on Dynamic Counting 
//  Framework without Randomized Sampling
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, without Randomized Sampling
//  2  Use Dynamic Counting Framework (QALSH) for P2PNNS
// -----------------------------------------------------------------------------
class NH_Counting_wo_S {
public:
    NH_Counting_wo_S(               // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        const float *data);             // input data

    // -------------------------------------------------------------------------
    ~NH_Counting_wo_S();            // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // collision threshold
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
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   nh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm of o' after transformation
    
    const float *data_;             // original data objects
    QALSH *lsh_;                    // QALSH for nh data

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const float *data,              // input data
        float &norm,                    // l2-norm-sqr of nh_data (return)
        float *nh_data);                // nh_data (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const float *query,             // input query
        float *nh_query);               // nh_query after transform (return)
};

} // end namespace p2h
