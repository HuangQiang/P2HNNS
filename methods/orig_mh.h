#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Orig_MH: Original Multilinear Hyperplane Hash
// -----------------------------------------------------------------------------
class Orig_MH {
public:
	Orig_MH(						// constructor
		int   n,						// number of input data 
		int   d,						// dimension of input data 
		int   M, 						// #proj vecotr used for a single hasher
		int   m,						// #single hasher of the compond hasher
		int   l,						// #hash tables
		const float *data);				// input data

	// -------------------------------------------------------------------------
    ~Orig_MH();						// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	int nns(						// point-to-hyperplane NNS
		int   top_k,					// top-k value
		int   cand,						// #candidates
		const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
	using SigType = int32_t;

	void get_sig_data(				// get the signature of data
		const float *data, 				// input data
		std::vector<SigType> &sig) const; // signature (return)

	void get_sig_query(				// get the signature of query
		const float *query,				// input query 
		std::vector<SigType> &sig) const; // signature (return)

	// -------------------------------------------------------------------------
	int64_t get_memory_usage()
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += buckets_.get_memory_usage(); // buckets_
		ret += SIZEFLOAT * projv_.capacity(); // projv
		return ret;
	}

protected:
    int   n_pts_;					// number of data objects
	int   dim_;						// dimension of data objects
    int   M_;						// #proj vector used for a single hasher
	int   m_;						// #single hasher of the compond hasher
	int   l_;						// #hash tables
	const float *data_;				// data objects

	std::vector<float> projv_;		// random projection vectors
	KLBucketingFlat<SigType> buckets_; // hash tables
};

} // end namespace p2h
