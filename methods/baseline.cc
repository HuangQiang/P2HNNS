#include "baseline.h"

namespace p2h {

// -----------------------------------------------------------------------------
Random_Scan::Random_Scan(           // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    const float *data)                  // input data
    : n_pts_(n), dim_(d), data_(data)
{
    // get random index by random shuffle
    random_id_ = new int[n];
    for (int i = 0; i < n; ++i) random_id_[i] = i;
    
    std::random_shuffle(random_id_, random_id_ + n);
}

// -----------------------------------------------------------------------------
Random_Scan::~Random_Scan()         // destructor
{
    delete[] random_id_; random_id_ = NULL;
}

// -----------------------------------------------------------------------------
void Random_Scan::display()         // display parameters
{
    printf("Parameters of Random_Scan:\n");
    printf("    n   = %d\n", n_pts_);
    printf("    dim = %d\n", dim_);
    printf("\n");
}

// -----------------------------------------------------------------------------
int Random_Scan::nns(               // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   nc,                           // number of total candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    int verif_cnt = std::min(nc + top_k - 1, n_pts_);
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = random_id_[i];
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    return verif_cnt;
}

// -----------------------------------------------------------------------------
Sorted_Scan::Sorted_Scan(           // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    const float *data)                  // input data
    : n_pts_(n), dim_(d), data_(data)            
{
    // sort data objects by their l2-norms in ascending order
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        arr[i].id_  = i;
        arr[i].key_ = sqrt(calc_inner_product(d-1, &data[i*d], &data[i*d]));
    }
    qsort(arr, n, sizeof(Result), ResultComp);

    // get the sorted order
    sorted_id_ = new int[n];
    for (int i = 0; i < n; ++i) sorted_id_[i] = arr[i].id_;

    // release space
    delete[] arr;
}

// -----------------------------------------------------------------------------
Sorted_Scan::~Sorted_Scan()         // destructor
{
    delete[] sorted_id_; sorted_id_ = NULL;
}

// -----------------------------------------------------------------------------
void Sorted_Scan::display()         // display parameters
{
    printf("Parameters of Sorted_Scan:\n");
    printf("    n   = %d\n", n_pts_);
    printf("    dim = %d\n", dim_);
    printf("\n");
}

// -----------------------------------------------------------------------------
int Sorted_Scan::nns(               // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   nc,                           // number of total candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    int verif_cnt = std::min(nc + top_k - 1, n_pts_);
    for (int i = 0; i < verif_cnt; ++i) {
        int   idx  = sorted_id_[i];
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
    }
    return verif_cnt;
}

// -----------------------------------------------------------------------------
Angular_Hash::Angular_Hash(         // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   M,                            // #proj vecotr used for a single hasher
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    float b,                            // interval ratio
    const float *data)                  // input data
    : n_pts_(n), dim_(d), M_(M), m_(m), l_(l), b_(b), data_(data)
{
    // -------------------------------------------------------------------------
    //  sort data objects by their l2-norms in ascending order
    // -------------------------------------------------------------------------
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        arr[i].id_  = i;
        arr[i].key_ = sqrt(calc_inner_product(d, &data[i*d], &data[i*d]));
    }
    qsort(arr, n, sizeof(Result), ResultComp);

    // -------------------------------------------------------------------------
    //  get the sorted id and sorted normalized data
    // -------------------------------------------------------------------------
    sorted_id_ = new int[n];
    float *sorted_data = new float[n * d];
    for (int i = 0; i < n; ++i) {
        int   idx  = arr[i].id_;
        float norm = arr[i].key_;

        sorted_id_[i] = idx;
        for (int j = 0; j < d; ++j) {
            sorted_data[i*d+j] = data[idx*d+j] / norm;
        }
    }

    // -------------------------------------------------------------------------
    //  divide datasets into blocks and build hash tables for each block
    // -------------------------------------------------------------------------
    int start = 0;
    while (start < n_pts_) {
        // partition block
        float max_radius  = arr[start].key_ / b;
        int   block_index = start, cnt = 0;
        while (block_index < n_pts_ && arr[block_index].key_ < max_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }
        // printf("%d: cnt = %d, start = %d\n", block_num_.size(), cnt, start);
        
        // add a block
        Basic_Hash *hash = NULL;
        if (M == 1) {
            hash = new EH_Hash(cnt, d, m, l, (const int*) sorted_id_+start, 
                (const float *) sorted_data+start*d);
        }
        else if (M == 2) {
            hash = new BH_Hash(cnt, d, m, l, (const int*) sorted_id_+start, 
                (const float *) sorted_data+start*d);
        }
        else {
            hash = new MH_Hash(cnt, d, M, m, l, (const int*) sorted_id_+start, 
                (const float *) sorted_data+start*d);
        }
        hash_.push_back(hash);
        block_num_.push_back(cnt);
        start += cnt;
    }
    assert(start == n_pts_);

    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] arr;
    delete[] sorted_data;
}

// -----------------------------------------------------------------------------
Angular_Hash::~Angular_Hash()       // destructor
{
    if (!hash_.empty()) {
        for (int i = 0; i < hash_.size(); ++i) {
            delete hash_[i]; hash_[i] = NULL;
        }
        hash_.clear(); hash_.shrink_to_fit();
    }
    delete[] sorted_id_; sorted_id_ = NULL;
    block_num_.clear(); block_num_.shrink_to_fit();
}

// -------------------------------------------------------------------------
void Angular_Hash::display()        // display parameters
{
    printf("Parameters of %s:\n", M_ > 2 ? "MH" : (M_ == 2 ? "BH" : "EH"));
    printf("    n       = %d\n",   n_pts_);
    printf("    dim     = %d\n",   dim_);
    printf("    M       = %d\n",   M_);
    printf("    m       = %d\n",   m_);
    printf("    l       = %d\n",   l_);
    printf("    b       = %.2f\n", b_);
    printf("    #blocks = %d\n\n", (int) block_num_.size());
}

// -----------------------------------------------------------------------------
int Angular_Hash::nns(              // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{    
    // -------------------------------------------------------------------------
    //  point-to-hyperplane NNS
    // -------------------------------------------------------------------------
    int n_cand = cand + top_k - 1;
    int verif_cnt = 0;

    for (int i = 0; i < block_num_.size(); ++i) {
        // check #candidates according to the ratio of block size
        int block_cand = (int) ceil((float) block_num_[i] * n_cand / n_pts_);
        block_cand = std::min(block_cand, n_cand - verif_cnt);
        verif_cnt += hash_[i]->nns(block_cand, data_, query, list);

        if (verif_cnt >= n_cand) break;
    }
    return verif_cnt;
}

} // end namespace p2h
