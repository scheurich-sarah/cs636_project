#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>

#include "kernel.h"

using std::cout;
using std::endl;
using namespace std::chrono;

int THD_COUNT = 1;

using std::string;


void _node2edge_attention(csr_t* snaph, array2d_t<float> & input_node_src, array2d_t<float> & input_node_dest, array2d_t<float> & output_edge_array){

    
    // get the offset and nebrs arrays
    vid_t* offset = snaph->get_offset_ptr();
    vid_t* nebrs = snaph->get_nebrs_ptr();
    
    for (int64_t i = 0; i<input_node_src.row_count; i++){

	    // get node degree
	    vid_t node_degree = snaph->get_degree(i);
	    // cout<< "node "<<i<<" degree = "<< node_degree<< endl;


	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];
	
		// perform src and dest concatenation

		// perform matrix mult on concat
	    }
    }

}

void _edge2node_backprop(csr_t* snaph, array2d_t<float> & input_edge, array2d_t<float> & output_node_src, array2d_t<float> & output_node_dest){


    
    // get the offset and nebrs arrays
    vid_t* offset = snaph->get_offset_ptr();
    vid_t* nebrs = snaph->get_nebrs_ptr();
    
    for (int64_t i = 0; i<input_edge.row_count; i++){

	    // get node degree
	    vid_t node_degree = snaph->get_degree(i);
	    // cout<< "node "<<i<<" degree = "<< node_degree<< endl;


	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];
	
		// split endege features into 2 nodes
	    }
    }

}

void invoke_forward_edge_attention(graph_t & graph, array2d_t<float> & feat_edge_src, array2d_t<float> & feat_edge_dest, array2d_t<float> & result) {
         return _node2edge_attention(&graph.csr, feat_edge_src, feat_edge_dest, result);
}

void invoke_backprop_attention(graph_t & graph, array2d_t<float> & feat_edge, array2d_t<float> & result1, array2d_t<float> & result2){
         return _edge2node_backprop(&graph.csr, feat_edge, result1, result2);
}


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    auto start = high_resolution_clock::now();
    //cout << "spmm " << op << ", reverse = " << reverse << endl;

    // input row count always = # nodes
    // input col count is different for based on layer type input or hidden
    // pubmed raw input features = 16, hidden input features = 3

    /*
    cout << "input row count = "<< input.row_count <<" col count = "
	    << input.col_count << endl;
    cout << "output row count = "<< output.row_count <<" col count = "
	    << output.col_count << endl;
    */

    /* 
    // print some sample values
    // output is same size as input but filled with 0's
    for (int64_t i = 0; i<3; i++){
	    for (int64_t j = 0; j<input.col_count; j++){
		cout<<"input in row, col "<< i << ",  "<<j<<
			" is "<< input.get_item(i, j)<<endl;
		cout<<"output in row, col "<< i << ",  "<<j<<
			" is "<< output.get_item(i, j)<<endl;
	    }
    }
    */
  
    // get the offset and nebrs arrays
    vid_t* offset = snaph->get_offset_ptr();
    vid_t* nebrs = snaph->get_nebrs_ptr();
    
    for (int64_t i = 0; i<input.row_count; i++){
	    /*
	    // view the row
	    array1d_t<float> node_row = input.get_row(i);
	    cout<<"\ninput for node "<< i <<" feat values: ";
	    for (int k=0; k<input.col_count; k++){
		    cout<< node_row[k]<<", ";
	    }	    
	    */

	    // get node degree
	    vid_t node_degree = snaph->get_degree(i);
	    // cout<< "node "<<i<<" degree = "<< node_degree<< endl;

	    // normalization procedure
	    // normalize arrays before summing if in backward pass
	    // normalize after summing all arrays if in forward pass
	    float pre_sum_norm; float post_sum_norm;
	    if (reverse){ // backward pass
		    post_sum_norm = (float) 1;
		    pre_sum_norm = (float) node_degree;
	    } else { // forward pass
		    pre_sum_norm = (float) 1;
		    post_sum_norm = (float) node_degree;
	    }

	    // pass pointer to input array row to the output array
	    // the GCN paper specifies that it adds self connections
	    // to the adjacency matrix, so we need to include the node
	    // in the sum
	    // memcpy(output[i], input[i], input.col_count*sizeof(input[i]));
	    for (int k = 0 ; k<input.col_count; k++){
		    output[i][k] = input[i][k]/pre_sum_norm;
	    }


	    /* old attempt to us op.h API
	    if (reverse){ // normalize prior to GAS
		    // use normalized row copy to modify pointer to output
		    // TODO don't we need to specify output[i] here
		    // when I try this, it breaks
		    output.row_copy_norm(input[i], i, node_degree);
		    
	    } else {
		    // use row copy to modify pointer to output
		    output.row_copy(input[i], i);
	    }
	    */

	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    // perform column wise add for all nebr features
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];
	
		/*
		// view output before adding nebr
		cout<<"\noutput = ";
		for (int k=0; k<input.col_count; k++){
			    cout<< output[i][k]<<", ";
		}*/

		/*
		// view nebr
		array1d_t<float> nebr_row = input.get_row((int64_t) neb);
		cout<<"\nnode "<<i<<" has nebr "<< neb << " with input feat ";
		for (int k=0; k<input.col_count; k++){
			    cout<< nebr_row[k]<<", ";
		}*/

		/* old attempt to use op.h API
		if (reverse){
			// TODO add normalized features
			// just regular add right now
			output.row_add(input[neb], j);
		} else {
			// add nebr features to current output features
			output.row_add(input[neb], j);
		}*/

		// loop through columns and add to output
		for (int64_t k = 0; k < input.col_count; k++){
			output[i][k] += input[neb][k]/pre_sum_norm;
		}
	    }

	    // do post-sum norm (will not change if in backward pass)
	    for (int k = 0 ; k<input.col_count; k++){
		    output[i][k] = output[i][k]/post_sum_norm;
	    }

	    /*
	    // view output row after adding all nebr
	    cout<<"\noutput post sum, norm  = "<<post_sum_norm<< " ";
	    for (int k=0; k<input.col_count; k++){
		    cout<< output[i][k]<<", ";
	    }
	    */
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "single thread gspmm pass takes " << duration.count() << endl;

}


// The signature for this function is in kernel.h
// very simple func,just pass a couple things
// one of which needs to be a pointer
// reverse: indicates whether or not to do forward or backward compute
void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
	 //cout<<"doing backpropagation"<< endl;
	 // backward computation uses csr
	 // normalzing involves dividing each input param/tensor
	 // by it's degree, which you need to calc
         return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
         //return _gspmm_mt(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
	 //cout<<"doing forward computation"<< endl;
	 // forward computation uses csc, the transpose of csr
	 // do GAS then normalize
         return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
         //return _gspmm_mt(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }

}



/*
void pick_largest_edge_weight(graph_t& graph,array2d_t<float> & feat, array2d_t<float> & result, bool inverse){

    // get the offset and nebrs arrays
    vid_t* offset = graph->get_offset_ptr();
    vid_t* nebrs = graph->get_nebrs_ptr();
    
    for (int64_t i = 0; i<feat.row_count; i++){

	    vid_t node_degree = graph->get_degree(i);

	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    result[i] = -std::numeric_limits<float>::infinity();
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];
		// keep the max edge weight
		if (feat[neb]> result[i]){
			result[i] = feat[neb];
		}

	    }
    }
}



void invoke_pick_largest_edge_weight(graph_t& graph,array2d_t<float> & feat, array2d_t<float> & result, bool inverse){
	if (inverse){
		return pick_largest_edge_weight(&graph.csr, feat, result, inverse)
	} else {
	
		return pick_largest_edge_weight(&graph.csc, feat, result, inverse)
	}
}


void invoke_subtract_max_score(graph_t& graph,array2d_t<float> &score_max, array2d_t<float> &feat, array2d_t<float> &result, bool inverse){


    // get the offset and nebrs arrays
    vid_t* offset = graph->get_offset_ptr();
    vid_t* nebrs = graph->get_nebrs_ptr();
    
    for (int64_t i = 0; i<feat.row_count; i++){
	    // get node degree
	    vid_t node_degree = graph->get_degree(i);

	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];

	
		// loop through columns and xxx
		for (int64_t k = 0; k < feat.col_count; k++){
			result[i][k] += input[neb][k];
		}
	    }
    }
}

void invoke_subtract_max_score(graph_t& graph,array2d_t<float> &score_max, array2d_t<float> &feat, array2d_t<float> &result, bool inverse){

	if (inverse){
		return subtract_max_score(&graph.csr, score_max, feat, result, inverse)
	} else {
	
		return subtract_max_score(&graph.csc, score_max, feat, result, inverse)
	}

}
        
void invoke_sum_scores_for_neighbors(graph_t& graph, array2d_t<float> & score, array2d_t<float> &result, bool inverse){


}

void invoke_sum_scores_for_neighbors(graph_t& graph, array2d_t<float> & score, array2d_t<float> &result, bool inverse){


	if (inverse){
		return sum_scores_for_neighbors(&graph.csr, score, result, inverse)
	} else {
	
		return sum_scores_for_neighbors(&graph.csc, score, result, inverse)
	}

}

void invoke_div_edge_score_by_neighborhood(graph_t& graph, array2d_t<float> & score_sum,  array2d_t<float> & score,  array2d_t<float> & result, bool inverse){


}


void invoke_div_edge_score_by_neighborhood(graph_t& graph, array2d_t<float> & score_sum,  array2d_t<float> & score,  array2d_t<float> & result, bool inverse){

	if (inverse){
		return div_edge_score_by_neighborhood(&graph.csr, score_sum, score, result, inverse)
	} else {
	
		return div_edge_score_by_neighborhood(&graph.csr, score_sum, score, result, inverse)
	}

}

void invoke_mult_edge_score_by_neighborhood(graph_t& graph, array2d_t<float> & accum, array2d_t<float> &  out,  array2d_t<float> & result, bool inverse){


}



void invoke_mult_edge_score_by_neighborhood(graph_t& graph, array2d_t<float> & accum, array2d_t<float> &  out,  array2d_t<float> & result, bool inverse){

	if (inverse){
		return mult_edge_score_by_neighborhood(&graph.csr, score_sum, score, result, inverse)
	} else {
	
		return mult_edge_score_by_neighborhood(&graph.csr, score_sum, score, result, inverse)
	}

}

*/
