#pragma once

#include "stdint.h"
using namespace std;

#ifdef B64
typedef uint64_t vid_t;
#elif B32
typedef uint32_t vid_t;
#endif

class csr_t {
 public:
    vid_t  v_count; //This is actual vcount in a graph 
    vid_t  e_count;
    vid_t  dst_size;
    vid_t* offset;
    vid_t* nebrs;
    int*  degrees;
    int64_t flag;

 public:
    csr_t() {};
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, int64_t a_flag, vid_t edge_count) {
        v_count = a_vcount;
        dst_size = a_dstsize;
        offset = (vid_t*)a_offset;
        nebrs = (vid_t*)a_nebrs;
        e_count = offset[edge_count];
        flag = a_flag;

	//cout << "csr.h, post csr_t.init, nebrs = "<< endl;
	//for (int i=0; i<e_count; i++){cout<<i<<": "<<nebrs[i]<<endl;}

    }
    vid_t get_vcount() {
        return v_count;
    }
    vid_t get_ecount() {
	// reminder: counts edges in both directions
        return e_count;
    }
    vid_t get_degree(vid_t index) {
        return offset[(int) index + 1] - offset[(int) index];
    }
   // both functions are options
   // not sure which is better style, keeping both
    vid_t* get_offset_ptr() {
	return offset;

    }
    vid_t get_offset(int vertex) {
	return offset[vertex];

    }

    vid_t* get_nebrs_ptr() {
        return nebrs;
    }
};

class edge_t {
 public:
     vid_t src;
     vid_t dst;
     //edge properties here if any.
};

class coo_t {
 public:
     edge_t* edges;
     vid_t dst_size;
     vid_t v_count;
     vid_t e_count;
     coo_t() {
         edges = 0;
         dst_size = 0;
         v_count = 0;
         e_count = 0;
     }
     void init(vid_t a_vcount, vid_t a_dstsize, vid_t a_ecount, edge_t* a_edges) {
     }
};

class graph_t {
 public:
    csr_t csr;
    csr_t csc;
    coo_t coo;
    bool mt;
 public:
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* a_offset1, void* a_nebrs1, int64_t flag, int64_t num_vcount) {

	// show offset nebrs for CSR and CSC formats
	// will be same for undirected graph
	/*
	cout<<"CSR format a_nebrs = "<<endl;
        vid_t* nebrs = (vid_t*)a_nebrs;
	for (int i=0; i<10; i++){cout<<i<<": "<<nebrs[i]<<endl;}
	cout<<"CSC format a_nebrs1 = "<<endl;
        vid_t* nebrs1 = (vid_t*)a_nebrs1;
	for (int i=0; i<10; i++){cout<<i<<": "<<nebrs1[i]<<endl;}
	*/

	// initialize csr and csc attributes of the graph	
    	csr.init(a_vcount, a_dstsize, a_offset, a_nebrs, flag, num_vcount);
	//cout << "csr.h, init csr attribute of graph_t "<< endl;

    	csc.init(a_vcount, a_dstsize, a_offset1, a_nebrs1, flag, num_vcount);
	//cout<<"car.h init csc attribute of graph_t"<<endl;
	
	mt = 0;
	cout<<"csr.h init mt attribute of graph_t = "<<mt<<endl;


    }

    vid_t get_vcount() {
	//cout << "called get_vcount in csr.h"<< endl;
        return csr.v_count;
    }
    vid_t get_edge_count() {
	//cout << "called get_edge_count in csr.h"<< endl;
        return csr.e_count;
    }
    bool get_mt() {
	//cout << "called get_edge_count in csr.h"<< endl;
        return mt;
    }
    void set_mt(bool b) {
	//cout << "called get_edge_count in csr.h"<< endl;
        mt = b;
    }
};

