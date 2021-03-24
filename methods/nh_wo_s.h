#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include "../lccs_bucket/bucketAlg/lcs_int.h"

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  NH_LCCS_wo_S: Nearest Hyperplane Hashing based on LCCS Bucketing without 
//  Randomized Sampling
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, without Randomized Sampling
//  2  Use LCCS Bucketing framework (LCCS-LSH) for P2PNNS
// -----------------------------------------------------------------------------
class NH_LCCS_wo_S {
public:
	using LCCS = mylccs::LCCS_SORT_INT;
	using SigType = int32_t;

	NH_LCCS_wo_S(					// constructor
		int   n,						// number of input data
		int   d,						// dimension of input data
		int   m,						// #hasher
		float w, 						// bucket width
		const float *data);				// input data

	// -------------------------------------------------------------------------
	virtual ~NH_LCCS_wo_S();		// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	virtual int nns(				// point-to-hyperplane NNS
		int   top_k,					// top-k value
		int   cand,						// #candidates
		const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
	virtual void get_sig_data(		// get the signature of data
		const float *data, 				// input data
		float norm,
		SigType *sig) const; 			// signature (return)

	// -------------------------------------------------------------------------
	virtual void get_sig_query(		// get the signature of query
		const float *query,				// input query 
		SigType *sig) const; 			// signature (return)

	// -------------------------------------------------------------------------
	virtual int64_t get_memory_usage() // get memory usage
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += sizeof(float)*projv_.capacity() + sizeof(float)*projb_.capacity() 
			+ sizeof(float)*norms_.capacity();
		ret += bucketerp_.get_memory_usage();
		ret += data_sigs_.get_memory_usage(); 
		return ret;
	}

protected:
	int   n_pts_;					// number of data objects
    int   dim_;						// dimension of data objects
    int   m_;						// #single hasher of the compond hasher
	float w_;						// bucket width
	const float *data_;				// input data objects

	std::vector<float> projv_;		// random projection vectors
	std::vector<float> projb_;		// random projection vectors
	std::vector<float> norms_;		// norm of transformed vectors

	NDArray<2, int32_t> data_sigs_;
	float max_norm_;
	LCCS bucketerp_; 				// hash tables

	// -------------------------------------------------------------------------
	void get_norms(const float* data);
};

} // end namespace p2h
