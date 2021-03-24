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
//  EH_Hash: Embedding Hyperplane Hash
// -----------------------------------------------------------------------------
class EH_Hash : public Basic_Hash {
public:
	EH_Hash(						// constructor
		int   n,						// number of input data
		int   d,						// dimension of input data
		int   m,						// #single hasher of the compond hasher
		int   l,						// #hash tables
		const int   *index,				// index of input data 
		const float *data);				// input data

	// -------------------------------------------------------------------------
    virtual ~EH_Hash();				// destructor

	// -------------------------------------------------------------------------
	virtual int nns(				// point-to-hyperplane NNS
		int   cand,						// #candidates
		const float *data,				// input data
        const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
    virtual void get_sig_data(		// get the signature of data
		const float *data, 				// input data
		std::vector<SigType> &sig) const; // signature (return)

	// -------------------------------------------------------------------------
    virtual void get_sig_query(		// get the signature of query
		const float *query,				// input query 
		std::vector<SigType> &sig) const; // signature (return)

	// -------------------------------------------------------------------------
	virtual int64_t get_memory_usage() // get memory usage
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += buckets_.get_memory_usage();
		ret += SIZEFLOAT * projv_.capacity(); // projv
		return ret;
	}

protected:
	int   n_pts_;					// number of data objects
    int   dim_;						// dimension of data objects
    int   m_;						// #single hasher of the compond hasher
	int   l_;						// #hash tables
    const int *index_;              // index of input data
    
    std::vector<float> projv_;		// random projection vectors
	KLBucketingFlat<SigType> buckets_; // hash tables
};

} // end namespace p2h
