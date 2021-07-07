#include "eh.h"

namespace p2h {

// -----------------------------------------------------------------------------
EH_Hash::EH_Hash(                   // constructor
    int   n,                            // number of input data
    int   d,                            // dimension of input data
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const int   *index,                 // index of input data
    const float *data)                  // input data
    : n_pts_(n), dim_(d), m_(m), l_(l), index_(index), buckets_(n, l)
{
    // sample random projection variables
    int size = m * l * d * d;
    projv_.resize(size);
    for (int i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
    }

    // build hash tables for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -------------------------------------------------------------------------
EH_Hash::~EH_Hash()                 // destructor
{
}

// -------------------------------------------------------------------------
void EH_Hash::get_sig_data(         // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                for (int j = 0; j < dim_; ++j) {
                    int pidx = (ll*m_+mm)*dim_*dim_ + i*dim_ + j;
                    projection += data[i] * data[j] * projv_[pidx];
                }
            }
            SigType new_sig = (projection > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -------------------------------------------------------------------------
void EH_Hash::get_sig_query(        // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l_
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                for (int j = 0; j < dim_; ++j) {
                    int pidx = (ll*m_+mm)*dim_*dim_ + i*dim_ + j;
                    projection += query[i] * query[j] * projv_[pidx];
                }
            }
            SigType new_sig = !(projection > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -------------------------------------------------------------------------
int EH_Hash::nns(                   // point-to-hyperplane NNS
    int   cand,                         // #candidates
    const float *data,                  // input data
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int verif_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx) {
        // verify the true distance of idx
        int   did  = index_[idx];
        float dist = fabs(calc_inner_product(dim_, &data[did*dim_], query));
        list->insert(dist, did + 1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
Orig_EH::Orig_EH(                   // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #single hasher of the compond hasher
    int   l,                            // #hash tables
    const float *data)                  // input data
    : n_pts_(n), dim_(d), m_(m), l_(l), data_(data), buckets_(n, l)
{
    // sample random projection variables
    int size = m * l * d * d;
    projv_.resize(size);
    for (int i = 0; i < size; ++i) {
        projv_[i] = gaussian(0.0f, 1.0f);
    }

    // build hash table for the hash values of data objects
    std::vector<SigType> sigs(l);
    for (int i = 0; i < n; ++i) {
        get_sig_data(&data[i*d], sigs);
        buckets_.insert(i, sigs);
    }
}

// -----------------------------------------------------------------------------
Orig_EH::~Orig_EH()                 // destructor
{
}

// -----------------------------------------------------------------------------
void Orig_EH::get_sig_data(         // get signature of data
    const float *data,                  // input data
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                for (int j = 0; j < dim_; ++j) {
                    int pidx = (ll*m_+mm)*dim_*dim_ + i*dim_ + j;
                    projection += data[i] * data[j] * projv_[pidx];
                }
            }
            SigType new_sig = (projection > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
void Orig_EH::get_sig_query(        // get signature of query
    const float *query,                 // input query
    std::vector<SigType> &sig) const    // signature (return)
{
    // the dimension of sig is l
    for (int ll = 0; ll < l_; ++ll) {
        SigType cur_sig = 0;
        for (int mm = 0; mm < m_; ++mm) {
            float projection = 0.0f;
            for (int i = 0; i < dim_; ++i) {
                for (int j = 0; j < dim_; ++j) {
                    int pidx = (ll*m_+mm)*dim_*dim_ + i*dim_ + j;
                    projection += query[i] * query[j] * projv_[pidx];
                }
            }
            SigType new_sig = !(projection > 0);
            cur_sig = (cur_sig<<1) | new_sig;
        }
        sig[ll] = cur_sig;
    }
}

// -----------------------------------------------------------------------------
void Orig_EH::display()             // display parameters
{
    printf("Parameters of Orig_EH:\n");
    printf("    n   = %d\n", n_pts_);
    printf("    dim = %d\n", dim_);
    printf("    m   = %d\n", m_);
    printf("    l   = %d\n", l_);
    printf("\n");
}

// -----------------------------------------------------------------------------
int Orig_EH::nns(                   // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    std::vector<SigType> sigs(l_);
    get_sig_query(query, sigs);
    
    int verif_cnt = 0;
    buckets_.for_cand(cand, sigs, [&](int idx){
        // verify the true distance of idx
        float dist = fabs(calc_inner_product(dim_, &data_[idx*dim_], query));
        list->insert(dist, idx + 1);
        ++verif_cnt;
    });
    return verif_cnt;
}

} // end namespace p2h
