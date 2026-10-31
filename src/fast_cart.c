#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define PRESORT_THRESH 4

typedef struct {
    float val;
    uint16_t idx;
    uint8_t c;
    uint8_t cnt;
} FastItem;

static inline void sort_fast_items(FastItem *arr, int n) {
    for (int i = 1; i < n; i++) {
        FastItem key = arr[i];
        int j = i - 1;
        while (j >= 0 && (arr[j].val > key.val || (arr[j].val == key.val && arr[j].idx > key.idx))) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

typedef struct {
    int max_nodes;
    int n_nodes;
    int K;
    int *feature;
    double *threshold;
    int *left_child;
    int *right_child;
    int *n_node_samples;
    double *impurity;
    double *values;
} TreeBuffer;

static int can_split_node(
    int depth,
    int max_depth,
    int min_samples_leaf,
    const int *counts,
    const double *w,
    const int *indices,
    const int *y,
    int n_unique,
    int K,
    TreeBuffer *tb,
    int node_id,
    double *out_total_w,
    int *out_total_samples,
    double *out_parent_sq_sum,
    double *class_w
) {
    double total_w = 0.0;
    int total_samples = 0;
    memset(class_w, 0, K * sizeof(double));

    for (int i = 0; i < n_unique; i++) {
        int idx = indices[i];
        double wi = w[idx];
        int ci = counts[idx];
        total_w += wi;
        total_samples += ci;
        class_w[y[idx]] += wi;
    }

    tb->n_node_samples[node_id] = total_samples;
    double *node_val = tb->values + (size_t)node_id * K;
    if (total_w > 0.0) {
        double inv_tw = 1.0 / total_w;
        for (int k = 0; k < K; k++) node_val[k] = class_w[k] * inv_tw;
    } else {
        for (int k = 0; k < K; k++) node_val[k] = 1.0 / K;
    }

    double parent_sq_sum = 0.0;
    int n_active_classes = 0;
    for (int k = 0; k < K; k++) {
        double cw = class_w[k];
        if (cw > 1e-15) {
            parent_sq_sum += cw * cw;
            n_active_classes++;
        }
    }
    double parent_gini = (total_w > 0.0) ? (1.0 - parent_sq_sum / (total_w * total_w)) : 0.0;
    tb->impurity[node_id] = parent_gini;
    tb->feature[node_id] = -1;
    tb->threshold[node_id] = 0.0;
    tb->left_child[node_id] = -1;
    tb->right_child[node_id] = -1;

    *out_total_w = total_w;
    *out_total_samples = total_samples;
    *out_parent_sq_sum = parent_sq_sum;

    if (n_active_classes <= 1 || parent_gini <= 1e-12) return 0;
    if (max_depth >= 0 && depth >= max_depth) return 0;
    if (total_samples < 2 * min_samples_leaf || n_unique < 2) return 0;
    return 1;
}

static void build_node_general(
    TreeBuffer *tb,
    int node_id,
    int depth,
    int max_depth,
    int min_samples_leaf,
    const float *X_T,
    int N_total,
    const double *w,
    const int *counts,
    int is_int_weights,
    const float *inv_int,
    const int *node_indices,
    const int *y,
    int n_unique,
    const FastItem *presorted_items, /* non-NULL only if n_unique >= PRESORT_THRESH */
    FastItem *scratch_pool,
    int D,
    int K,
    const int *feat_order
) {
    double total_w = 0.0;
    int total_samples = 0;
    double parent_sq_sum = 0.0;
    double class_w[64];

    if (!can_split_node(depth, max_depth, min_samples_leaf, counts, w, node_indices, y, n_unique, K, tb, node_id,
                        &total_w, &total_samples, &parent_sq_sum, class_w)) {
        return;
    }

    int n_active_classes = 0;
    int class_cnt[64];
    int m1 = 0, m2 = 0, min_c = 1000000;
    for (int k = 0; k < K; k++) {
        int cc = (int)(class_w[k] + 0.5);
        class_cnt[k] = cc;
        if (class_w[k] > 1e-15) {
            n_active_classes++;
            if (cc < min_c) min_c = cc;
            if (cc > m1) { m2 = m1; m1 = cc; }
            else if (cc > m2) { m2 = cc; }
        }
    }

    int best_f = -1;
    double best_t = 0.0;
    FastItem local_items[64];

    if (is_int_weights) {
        int parent_sq_int = (int)(parent_sq_sum + 0.5);
        float best_score = (float)parent_sq_int * inv_int[total_samples] + 1e-6f;
        float max_possible_score = (float)total_samples - 1e-5f;
        /* General tight upper bound on binary split score:
           In any binary split of integer class counts with max class count m1,
           if n_active_classes == n_unique and min_samples_leaf == 1: */
        if (n_active_classes == n_unique && total_samples == n_unique && min_samples_leaf == 1) {
            max_possible_score = 2.0f - 1e-5f;
        }

        int left_cnt[64];
        memset(left_cnt, 0, K * sizeof(int));
        uint8_t touched[256];

        for (int fi = 0; fi < D; fi++) {
            int f = feat_order[fi];
            const FastItem *items;
            if (presorted_items != NULL) {
                items = presorted_items + (size_t)fi * n_unique;
            } else {
                const float *xf = X_T + (size_t)f * N_total;
                for (int i = 0; i < n_unique; i++) {
                    int idx = node_indices[i];
                    local_items[i].val = xf[idx];
                    local_items[i].idx = (uint16_t)idx;
                    local_items[i].c = (uint8_t)y[idx];
                    local_items[i].cnt = (uint8_t)counts[idx];
                }
                sort_fast_items(local_items, n_unique);
                items = local_items;
            }

            if (items[0].val + 1e-7f >= items[n_unique - 1].val) continue;

            int sq_left = 0;
            int sq_right = parent_sq_int;
            int cnt_left = 0;
            int n_touched = 0;

            memset(left_cnt, 0, K * sizeof(int));
            for (int i = 0; i < n_unique - 1; i++) {
                uint8_t c = items[i].c;
                int ci = (int)items[i].cnt;
                int lc = left_cnt[c];
                int lc_new = lc + ci;
                left_cnt[c] = lc_new;
                cnt_left += ci;
                sq_left += (2 * lc + ci) * ci;
                sq_right -= (2 * (class_cnt[c] - lc_new) + ci) * ci;

                int cnt_right = total_samples - cnt_left;
                float v_curr = items[i].val;
                float v_next = items[i + 1].val;
                if (v_next > v_curr + 1e-7f && cnt_left >= min_samples_leaf && cnt_right >= min_samples_leaf) {
                    float score = (float)sq_left * inv_int[cnt_left] + (float)sq_right * inv_int[cnt_right];
                    if (score > best_score + 1e-6f) {
                        best_score = score;
                        best_f = f;
                        best_t = 0.5 * ((double)v_curr + (double)v_next);
                    }
                }
            }
            if (best_score >= max_possible_score) break;
        }
    } else {
        double best_score = parent_sq_sum / total_w + 1e-12;
        double max_possible_score = total_w - 1e-12;
        double left_w[64];
        memset(left_w, 0, K * sizeof(double));
        uint8_t touched[256];

        for (int fi = 0; fi < D; fi++) {
            int f = feat_order[fi];
            const FastItem *items;
            if (presorted_items != NULL) {
                items = presorted_items + (size_t)fi * n_unique;
            } else {
                const float *xf = X_T + (size_t)f * N_total;
                for (int i = 0; i < n_unique; i++) {
                    int idx = node_indices[i];
                    local_items[i].val = xf[idx];
                    local_items[i].idx = (uint16_t)idx;
                    local_items[i].c = (uint8_t)y[idx];
                    local_items[i].cnt = (uint8_t)counts[idx];
                }
                sort_fast_items(local_items, n_unique);
                items = local_items;
            }

            if (items[0].val + 1e-7f >= items[n_unique - 1].val) continue;

            double w_left = 0.0;
            double sq_left = 0.0;
            double sq_right = parent_sq_sum;
            int cnt_left = 0;
            int n_touched = 0;

            for (int i = 0; i < n_unique - 1; i++) {
                int idx = items[i].idx;
                uint8_t c = items[i].c;
                int ci = (int)items[i].cnt;
                double wi = w[idx];

                double lw_c = left_w[c];
                if (lw_c == 0.0) touched[n_touched++] = c;
                sq_left += (2.0 * lw_c + wi) * wi;
                double lw_new = lw_c + wi;
                left_w[c] = lw_new;
                w_left += wi;
                cnt_left += ci;

                double r_new = class_w[c] - lw_new;
                sq_right -= (2.0 * r_new + wi) * wi;

                int cnt_right = total_samples - cnt_left;
                if (cnt_left < min_samples_leaf || cnt_right < min_samples_leaf) continue;

                float v_curr = items[i].val;
                float v_next = items[i + 1].val;
                if (v_next <= v_curr + 1e-7f) continue;

                double w_right = total_w - w_left;
                if (w_left <= 1e-15 || w_right <= 1e-15) continue;

                double score = sq_left / w_left + sq_right / w_right;
                if (score > best_score + 1e-12) {
                    best_score = score;
                    best_f = f;
                    best_t = 0.5 * ((double)v_curr + (double)v_next);
                }
            }
            for (int ti = 0; ti < n_touched; ti++) {
                left_w[touched[ti]] = 0.0;
            }
            if (best_score >= max_possible_score) break;
        }
    }

    if (best_f == -1) return;

    unsigned char is_left[512];
    memset(is_left, 0, N_total * sizeof(unsigned char));
    int left_indices[512], right_indices[512];
    int n_left_unique = 0, n_right_unique = 0;

    const float *best_xf = X_T + (size_t)best_f * N_total;
    for (int i = 0; i < n_unique; i++) {
        int idx = node_indices[i];
        if ((double)best_xf[idx] <= best_t) {
            is_left[idx] = 1;
            left_indices[n_left_unique++] = idx;
        } else {
            right_indices[n_right_unique++] = idx;
        }
    }

    if (n_left_unique == 0 || n_right_unique == 0) return;

    int left_id = tb->n_nodes++;
    int right_id = tb->n_nodes++;
    tb->feature[node_id] = best_f;
    tb->threshold[node_id] = best_t;
    tb->left_child[node_id] = left_id;
    tb->right_child[node_id] = right_id;

    double dummy_tw, dummy_sq, dummy_cw[64];
    int dummy_ts;
    int split_left = can_split_node(depth + 1, max_depth, min_samples_leaf, counts, w, left_indices, y, n_left_unique, K, tb, left_id,
                                    &dummy_tw, &dummy_ts, &dummy_sq, dummy_cw);
    int split_right = can_split_node(depth + 1, max_depth, min_samples_leaf, counts, w, right_indices, y, n_right_unique, K, tb, right_id,
                                     &dummy_tw, &dummy_ts, &dummy_sq, dummy_cw);

    if (!split_left && !split_right) return;

    int keep_presort_l = (split_left && presorted_items != NULL && n_left_unique >= PRESORT_THRESH);
    int keep_presort_r = (split_right && presorted_items != NULL && n_right_unique >= PRESORT_THRESH);

    FastItem *items_left = keep_presort_l ? scratch_pool : NULL;
    FastItem *items_right = keep_presort_r ? (scratch_pool + (keep_presort_l ? (size_t)D * n_left_unique : 0)) : NULL;
    FastItem *next_scratch = scratch_pool + (keep_presort_l ? (size_t)D * n_left_unique : 0) + (keep_presort_r ? (size_t)D * n_right_unique : 0);

    if (keep_presort_l && keep_presort_r) {
        for (int fi = 0; fi < D; fi++) {
            const FastItem *src = presorted_items + (size_t)fi * n_unique;
            FastItem *dst_l = items_left + (size_t)fi * n_left_unique;
            FastItem *dst_r = items_right + (size_t)fi * n_right_unique;
            int il = 0, ir = 0;
            for (int i = 0; i < n_unique; i++) {
                FastItem it = src[i];
                if (is_left[it.idx]) dst_l[il++] = it;
                else dst_r[ir++] = it;
            }
        }
    } else if (keep_presort_l) {
        for (int fi = 0; fi < D; fi++) {
            const FastItem *src = presorted_items + (size_t)fi * n_unique;
            FastItem *dst_l = items_left + (size_t)fi * n_left_unique;
            int il = 0;
            for (int i = 0; i < n_unique; i++) {
                FastItem it = src[i];
                if (is_left[it.idx]) dst_l[il++] = it;
            }
        }
    } else if (keep_presort_r) {
        for (int fi = 0; fi < D; fi++) {
            const FastItem *src = presorted_items + (size_t)fi * n_unique;
            FastItem *dst_r = items_right + (size_t)fi * n_right_unique;
            int ir = 0;
            for (int i = 0; i < n_unique; i++) {
                FastItem it = src[i];
                if (!is_left[it.idx]) dst_r[ir++] = it;
            }
        }
    }

    if (split_left) {
        build_node_general(
            tb, left_id, depth + 1, max_depth, min_samples_leaf,
            X_T, N_total, w, counts, is_int_weights, inv_int, left_indices, y, n_left_unique,
            items_left, next_scratch, D, K, feat_order
        );
    }
    if (split_right) {
        build_node_general(
            tb, right_id, depth + 1, max_depth, min_samples_leaf,
            X_T, N_total, w, counts, is_int_weights, inv_int, right_indices, y, n_right_unique,
            items_right, next_scratch, D, K, feat_order
        );
    }
}

static FastItem *global_pools[8] = {NULL};

int fit_tree_general_c(
    int thread_id,
    const float *X_T,
    int N_total,
    const int *y,
    const double *w,
    const int *counts,
    int is_int_weights,
    const int *root_indices,
    int n_unique,
    const int *root_presorted_idx, /* (D, N_total) */
    int D,
    int K,
    int max_depth,
    int min_samples_leaf,
    const int *feat_order,
    int max_nodes,
    int *out_feature,
    double *out_threshold,
    int *out_left,
    int *out_right,
    int *out_n_samples,
    double *out_impurity,
    double *out_values
) {
    float inv_int[1024];
    inv_int[0] = 0.0f;
    for (int i = 1; i < 1024; i++) inv_int[i] = 1.0f / (float)i;

    if (thread_id < 0 || thread_id >= 8) thread_id = 0;
    if (global_pools[thread_id] == NULL) {
        global_pools[thread_id] = (FastItem*)malloc((size_t)24 * 4096 * 256 * sizeof(FastItem));
    }
    FastItem *pool = global_pools[thread_id];
    FastItem *root_items = NULL;
    FastItem *scratch_pool = pool;

    int active_feat_order[4096];
    int D_active = 0;
    if (n_unique >= PRESORT_THRESH) {
        root_items = pool;
        unsigned char active[512];
        memset(active, 0, N_total * sizeof(unsigned char));
        for (int i = 0; i < n_unique; i++) active[root_indices[i]] = 1;

        for (int fi = 0; fi < D; fi++) {
            int f = feat_order[fi];
            const float *xf = X_T + (size_t)f * N_total;
            const int *g_idx = root_presorted_idx + (size_t)f * N_total;
            if (xf[g_idx[0]] + 1e-7f >= xf[g_idx[N_total - 1]]) continue;
            FastItem *dst = root_items + (size_t)D_active * n_unique;
            int pos = 0;
            for (int i = 0; i < N_total; i++) {
                int idx = g_idx[i];
                if (active[idx]) {
                    FastItem it;
                    it.val = xf[idx];
                    it.idx = (uint16_t)idx;
                    it.c = (uint8_t)y[idx];
                    it.cnt = (uint8_t)counts[idx];
                    dst[pos++] = it;
                }
            }
            if (dst[0].val + 1e-7f < dst[n_unique - 1].val) {
                active_feat_order[D_active++] = f;
            }
        }
        scratch_pool = pool + (size_t)D_active * n_unique;
    } else {
        for (int fi = 0; fi < D; fi++) active_feat_order[D_active++] = feat_order[fi];
    }

    TreeBuffer tb;
    tb.max_nodes = max_nodes;
    tb.n_nodes = 1;
    tb.K = K;
    tb.feature = out_feature;
    tb.threshold = out_threshold;
    tb.left_child = out_left;
    tb.right_child = out_right;
    tb.n_node_samples = out_n_samples;
    tb.impurity = out_impurity;
    tb.values = out_values;

    build_node_general(
        &tb, 0, 0, max_depth, min_samples_leaf,
        X_T, N_total, w, counts, is_int_weights, inv_int, root_indices, y, n_unique,
        root_items, scratch_pool, D_active, K, active_feat_order
    );
    return tb.n_nodes;
}

void fit_ensemble_c(
    int B,
    const float *X_T,
    int N_total,
    const int *y,
    const double *all_w,      /* (B, N_total) */
    const int *all_counts,    /* (B, N_total) */
    int is_int_weights,
    const int *all_root_idx,  /* (B, N_total) */
    const int *all_n_unique,  /* (B,) */
    const int *root_presorted_idx, /* (D, N_total) */
    int D,
    int K,
    int max_depth,
    int min_samples_leaf,
    const int *all_feat_order, /* (B, D) */
    int max_nodes,
    int *out_n_nodes,         /* (B,) */
    int *out_feature,         /* (B, max_nodes) */
    double *out_threshold,    /* (B, max_nodes) */
    int *out_left,            /* (B, max_nodes) */
    int *out_right,           /* (B, max_nodes) */
    int *out_n_samples,       /* (B, max_nodes) */
    double *out_impurity,     /* (B, max_nodes) */
    double *out_values        /* (B, max_nodes, K) */
) {
    int max_t = omp_get_max_threads();
    if (max_t > 8) max_t = 8;
    for (int t = 0; t < max_t; t++) {
        if (global_pools[t] == NULL) {
            global_pools[t] = (FastItem*)malloc((size_t)24 * 4096 * 256 * sizeof(FastItem));
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < B; b++) {
        int tid = omp_get_thread_num();
        out_n_nodes[b] = fit_tree_general_c(
            tid,
            X_T, N_total, y,
            all_w + (size_t)b * N_total,
            all_counts + (size_t)b * N_total,
            is_int_weights,
            all_root_idx + (size_t)b * N_total,
            all_n_unique[b],
            root_presorted_idx,
            D, K, max_depth, min_samples_leaf,
            all_feat_order + (size_t)b * D,
            max_nodes,
            out_feature + (size_t)b * max_nodes,
            out_threshold + (size_t)b * max_nodes,
            out_left + (size_t)b * max_nodes,
            out_right + (size_t)b * max_nodes,
            out_n_samples + (size_t)b * max_nodes,
            out_impurity + (size_t)b * max_nodes,
            out_values + (size_t)b * max_nodes * K
        );
    }
}

void predict_ensemble_c(
    int B,
    int max_nodes,
    int K,
    const int *out_f,       /* (B, max_nodes) */
    const double *out_t,    /* (B, max_nodes) */
    const int *out_l,       /* (B, max_nodes) */
    const int *out_r,       /* (B, max_nodes) */
    const double *out_val,  /* (B, max_nodes, K) */
    const float *X,         /* (N_eval, D) */
    int N_eval,
    int D,
    int *tree_preds,        /* (B, N_eval) */
    double *vote_probs      /* (N_eval, K) */
) {
    memset(vote_probs, 0, (size_t)N_eval * K * sizeof(double));
    double inv_B = 1.0 / (double)B;

    for (int b = 0; b < B; b++) {
        const int *fb = out_f + (size_t)b * max_nodes;
        const double *tb = out_t + (size_t)b * max_nodes;
        const int *lb = out_l + (size_t)b * max_nodes;
        const int *rb = out_r + (size_t)b * max_nodes;
        const double *vb = out_val + (size_t)b * max_nodes * K;
        int *tp_b = tree_preds + (size_t)b * N_eval;

        for (int i = 0; i < N_eval; i++) {
            const float *xi = X + (size_t)i * D;
            int node = 0;
            while (fb[node] != -1) {
                if ((double)xi[fb[node]] <= tb[node]) {
                    node = lb[node];
                } else {
                    node = rb[node];
                }
            }
            const double *v_leaf = vb + (size_t)node * K;
            int best_k = 0;
            double best_v = v_leaf[0];
            for (int k = 1; k < K; k++) {
                if (v_leaf[k] > best_v) {
                    best_v = v_leaf[k];
                    best_k = k;
                }
            }
            tp_b[i] = best_k;
            vote_probs[i * K + best_k] += inv_B;
        }
    }
}
