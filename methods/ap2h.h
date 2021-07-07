#pragma once

#include "baseline.h"
#include "fh.h"
#include "nh.h"
#include "nh_counting.h"

namespace p2h {

// -----------------------------------------------------------------------------
int linear_scan(                    // P2HNNS using Linear_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int random_scan(                    // P2HNNS using Random_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int sorted_scan(                    // P2HNNS using Sorted_Scan
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int embed_hash(                     // P2HNNS using EH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int bilinear_hash(                  // P2HNNS using BH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int multilinear_hash(               // P2HNNS using MH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int fh(                             // P2HNNS using FH (Furthest_Hash)
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int fh_minus(                       // P2HNNS using FH_Minus
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int nh_lccs(                        // P2HNNS using NH (Nearest_Hash)
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    float w,                            // bucket width
    float s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int nh_counting(                    // P2HNNS using NH_Counting
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int fh_wo_sampling(                 // P2HNNS using FH_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int fh_minus_wo_sampling(           // P2HNNS using FH_Minus_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int nh_lccs_wo_sampling(            // P2HNNS using NH_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    float w,                            // bucket width
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int nh_counting_wo_sampling(        // P2HNNS using NH_Counting_wo_S
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int orig_embed_hash(                // P2HNNS using Orig_EH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int orig_bilinear_hash(             // P2HNNS using Orig_BH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

// -----------------------------------------------------------------------------
int orig_multilinear_hash(          // P2HNNS using Orig_MH
    int   n,                            // number of data  objects
    int   qn,                           // number of query objects
    int   d,                            // dimension of space
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *output_folder,          // output folder
    const float *data,                  // data set
    const float *query,                 // query set
    const Result *R);                   // truth set

} // end namespace p2h
