#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "rqalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Furthest_Hash (FH): Furthest Hyperplane Hash
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, with Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. With Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
class Furthest_Hash {
public:
	Furthest_Hash(					// constructor
		int   n,						// number of data objects
		int   d,						// dimension of data objects
		int   m,						// #hash tables
		int   s, 						// scale factor of dimension
		float b,						// interval ratio
		const float *data);				// input data

	// -------------------------------------------------------------------------
	~Furthest_Hash();				// destructor

	// -------------------------------------------------------------------------
	void display();					// display parameters

	// -------------------------------------------------------------------------
	int nns(						// point-to-hyperplane NNS
		int   top_k,					// top-k value
		int   l,						// separation threshold
		int   cand,						// #candidates
		const float *query,				// input query
		MinK_List *list);				// top-k results (return)

	// -------------------------------------------------------------------------
	int64_t get_memory_usage()		// get memory usage
	{
		int64_t ret = 0;
		ret += sizeof(*this);
		ret += SIZEFLOAT * fh_dim_; // centroid_
		ret += SIZEINT * n_pts_;	// shift_id_
		for (auto hash : hash_) { 	// blocks_
			ret += hash->get_memory_usage();
		}
		return ret;
	}

protected:
    int   n_pts_;					// number of data objects
	int   dim_;						// dimension of data objects
	int   scale_;					// scale factor of dimension
	int   sample_dim_;				// sample dimension
	int   fh_dim_;					// new data dimension after transformation
	float b_;						// interval ratio
	float M_;						// max l2-norm sqr of o'
	const float *data_;				// original data objects

	int *shift_id_;					// shift data id
	int block_cand_ = 10000;		// max #candidates checked in each block
	std::vector<RQALSH*> hash_;		// blocks

	// -------------------------------------------------------------------------
	void transform_data(			// data transformation
		const  float *data,				// input data
		float  *prob,					// probability vector
		bool   *checked,				// is checked?
		float  &norm,					// norm of fh_data (return)
		int    &sample_d,				// sample dimension (return)
		Result *sample_data,			// sample data (return)
		float  *centroid,				// centroid (return)
		int    *ctrd_cnt);				// centroid coordinate conuter (return)

	// -------------------------------------------------------------------------
	int sampling(					// sampling coordinate based on prob
		int   d,						// dimension
		const float *prob);				// input probability

	// -------------------------------------------------------------------------
	float calc_transform_dist(		// calc l2-dist-sqr after transformation 
		int   sample_d,					// dimension of sample data
		const Result *sample_data,		// sample data
		const float *centroid);			// centroid after data transformation

	// -------------------------------------------------------------------------
	void transform_query(			// query transformation
		const  float *query,			// input query
		float  &norm_sample_q,			// l2-norm sqr of q after transform (return)
		int    &sample_d,				// dimension of sample query (return)
		Result *sample_query);			// sample query after transform (return)
};

} // end namespace p2h
