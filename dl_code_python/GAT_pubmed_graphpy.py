import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
import datetime

import pygraph as gone
import gat
import pubmed_util

import create_graph as cg


if __name__ == "__main__":
    ingestion_flag = gone.enumGraph.eUdir
    ifile = "/home/pkumar/data/pubmed/graph_structure"
    num_vcount = 19717
    
    # data structure required for implementation
    edge_index = []
    G = cg.create_csr_graph_simple(ifile, num_vcount, ingestion_flag, edge_index)
    #print('GAT_pubmed edge_index = ', edge_index)
    
    # specific to pubmed
    input_feature_dim = 500
    
    # GAT is the type of the model
    # it's defined in GAT.py and inherits from Pytorch NNs Module
    # Model needs details like:
    # graph
    # num_of_layers (must be length of lists)
    # num_heads_per_layer (list of length num_layers)
    # num_features_per_layer (list of length num_layers + 1, has extra param for output layer features, which is 1, the predicted class) 
    # add_skip_connection (default=True) 
    # bias (default=True)
    # dropout (default=0.6)
    # log_attention_weights (default=False)
    
    # gat_net is the name of the model you can work with
    gat_net = gat.GAT(G, 2, [1,1], [500, 8, 1], False, True, 0.6)

    # extract geatures from the pubmed file
    feature = pubmed_util.read_feature_info("/home/pkumar/data/pubmed/feature/feature.txt")
    # create test and training labels
    train_id = pubmed_util.read_index_info("/home/pkumar/data/pubmed/index/train_index.txt")
    test_id = pubmed_util.read_index_info("/home/pkumar/data/pubmed/index/test_index.txt")
    test_y_label =  pubmed_util.read_label_info("/home/pkumar/data/pubmed/label/test_y_label.txt")
    train_y_label =  pubmed_util.read_label_info("/home/pkumar/data/pubmed/label/y_label.txt")
    

    # convert extracted features to tensors
    feature = torch.tensor(feature)

    train_id = torch.tensor(train_id)
    test_id = torch.tensor(test_id)
    train_y_label = torch.tensor(train_y_label)
    test_y_label = torch.tensor(test_y_label)


    # label_set = set(labels)
    # class_label_list = []
    # for each in label_set:
    #     class_label_list.append(each)
    # labeled_nodes_train = torch.tensor(train_idx)  # only the instructor and the president nodes are labeled
    # # labels_train = torch.tensor(output_train_label_encoded )  # their labels are different
    # labeled_nodes_test = torch.tensor(test_idx)  # only the instructor and the president nodes are labeled
    # # labels_test = torch.tensor(output_test_label_encoded)  # their labels are different

    # train the network
    # GAT uses Adam optimization
    optimizer = torch.optim.Adam(itertools.chain(gat_net.parameters()), lr=0.01, weight_decay=5e-4)
    all_logits = []
    start = datetime.datetime.now()

    for epoch in range(2):
        # provide data to the model that is required by forward method
        logits = gat_net(feature)
        #print ('check result')
        #print(logits)
        #print(logits.size())
        # get logits output probability
        all_logits.append(logits.detach())
        # take softmax to predict one answer
        logp = F.log_softmax(logits, 1)
        #print("prediction",logp[train_id])
    
        # commpute the negative log likelihood loss for classification
        #print('loss_size', logp[train_id].size(), train_y_label.size())
        loss = F.nll_loss(logp[train_id], train_y_label)

        #print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # check the accuracy for test data
        #logits_test = net.forward(feature)
        #logp_test = F.log_softmax(logits_test, 1)

        #acc_val = pubmed_util.accuracy(logp_test[test_id], test_y_label)
        #print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

    end = datetime.datetime.now()
    difference = end - start
    print("the time of graphpy is:", difference)
    logits_test = net.forward(feature)
    logp_test = F.log_softmax(logits_test, 1)
    acc_val = pubmed_util.accuracy(logp_test[test_id], test_y_label)
    print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

