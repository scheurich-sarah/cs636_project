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

    //The core logic goes here.    
    // needs to be independent of any tensor data struc

    // GAS in this case: gather all neighbors' messgages/tensors, then sum
    // each vertex has a feature vec, when you sum it, you get another vec

    
    // The GCN paper describes adding self connections to the adjacency matrix
    // this means we also need to add the current node's message/tensor
    
    // If in backward, normalize it first, else normalize it after computation
}

// multi threaded version
void _gspmm_mt(csr_t* snaph, array2d_t<float> & input, 
		array2d_t<float> & output, op_t op,
		bool reverse, bool norm /*= true*/) {

    auto start_mt = high_resolution_clock::now();
    // get the offset and nebrs arrays
    vid_t* offset = snaph->get_offset_ptr();
    vid_t* nebrs = snaph->get_nebrs_ptr();
    
    #pragma omp parallel for
    for (int64_t i = 0; i<input.row_count; i++){

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

	    // get node neighbors
	    vid_t start_idx = offset[i];
	    vid_t end_idx = offset[i + 1];
	
	    // loop through neighbors
	    // perform column wise add for all nebr features
	    // can loop through all neighbors in parallel

	    //#pragma omp parallel for
	    for (int64_t j= start_idx; j < end_idx; j++) {
		vid_t neb = nebrs[j];
	

		// loop through columns and add to output
		for (int64_t k = 0; k < input.col_count; k++){
			output[i][k] += input[neb][k]/pre_sum_norm;
		}
	    }

	    // do post-sum norm (will not change if in backward pass)
	    for (int k = 0 ; k<input.col_count; k++){
		    output[i][k] = output[i][k]/post_sum_norm;
	    }

    }
    auto stop_mt = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop_mt-start_mt);
    cout << "multi thread gspmm pass takes " << duration.count() << endl;

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

void invoke_gspmm_mt(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
	 //cout<<"doing backpropagation"<< endl;
	 // backward computation uses csr
	 // normalzing involves dividing each input param/tensor
	 // by it's degree, which you need to calc
         return _gspmm_mt(&graph.csr, input_array,
			 output_array, eSUM, reverse, norm);
    } else {
	 //cout<<"doing forward computation"<< endl;
	 // forward computation uses csc, the transpose of csr
	 // do GAS then normalize
         return _gspmm_mt(&graph.csc, input_array,
			 output_array, eSUM, reverse, norm);
    }

}
