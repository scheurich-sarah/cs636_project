#pragma once

#include "csr.h"
#include "op.h"

extern int THD_COUNT;
    
void invoke_gspmm(graph_t& graph, array2d_t<float> & input, array2d_t<float> & output, 
                 bool reverse, bool norm);

void invoke_forward_edge_attention(graph_t & graph, array2d_t<float> & feat_edge_src, array2d_t<float> & feat_edge_dest, array2d_t<float> & result);

void invoke_backprop_attention(graph_t & graph, array2d_t<float> & feat_edge, array2d_t<float> & result1, array2d_t<float> & result2);

/*void invoke_pick_largest_edge_weight(graph_t& graph,array2d_t<float> & feat, array2d_t<float> & result, bool inverse);

void invoke_subtract_max_score(graph_t& graph,array2d_t<float> &score_max, array2d_t<float> &feat, array2d_t<float> &result, bool inverse);
        
void invoke_sum_scores_for_neighbors(graph_t& graph, array2d_t<float> & score, array2d_t<float> &result, bool inverse);

void invoke_div_edge_score_by_neighborhood(graph_t& graph,score_sum,  array2d_t<float> & score,  array2d_t<float> & result, bool inverse):

void invoke_mult_edge_score_by_neighborhood(graph_t& graph,accum, array2d_t<float> &  out,  array2d_t<float> & result, bool inverse):
*/
