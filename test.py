from __future__ import print_function
import numpy as np
from sklearn import preprocessing
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#import random
try: import cPickle as pickle
except: import pickle
from math import ceil
from math import sqrt
try: from pyflann import *
except: pass
try: from pykdtree.kdtree import KDTree
except: pass
try: from sklearn.neighbors import NearestNeighbors
except: pass
try: from annoy import AnnoyIndex
except: pass
try: from scipy.spatial import cKDTree
except: pass
try: import nmslib
except: pass



def get_data_process(li, list_of_index):
    result = []
    for entry in li:
        temp = ()
        for index in list_of_index:
            temp += (entry[index],)
        result.append(temp)
    return np.asarray(result)
    

def remove_list_from_list(a,b):
    return list(set(a)-set(b))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_array_based_on_index(array, index_list):
    return np.asarray([array[i] for i in index_list])

def get_subsampling_index2(data_process, standard_scale = True, cutoff_sig = 0.02, rate = 0.3, \
                           method = "pykdtree", verbose = True):
    
    print("Started NN-subsampling, original length: {}".format(len(data_process)))
    
    method = method.lower()
    start = time.time()

    if method == "flann":
        print("use flann backend")
    elif method == "pykdtree":
        print("use pykdtree backend")
    elif method == "sklearn":
        print("use slearn nearest neighbors backend")
    elif method == "scipy":
        print("use scipy cKDTree backend")
    elif method == "annoy":
        print("use annoy backend")
    elif method == "nmslib":
        print("use nmslib backend")
    else:
        print("method {} not impletemented".format(method))
        raise NotImplemented
    
    if standard_scale:
        if verbose:
            print("Subample with standard scaled data")
        data_process = StandardScaler().fit_transform(np.asarray(data_process).copy())
    else:
        if verbose:
            print("Subample with original data")
        data_process = np.asarray(data_process).copy()
    

    list_of_descs = zip(*data_process)
    sum_std2 = 0.    
    for descs in list_of_descs:
        temp_std = np.std(descs)
        sum_std2 += temp_std**2


    #setting cutoff distance
    cutoff = cutoff_sig * np.sqrt(sum_std2)
      
    overall_keep_list = np.arange(len(data_process)).tolist() 
    
    keep_going = True
    while keep_going:
        if verbose:
            print('start total length: {}'.format(len(overall_keep_list)))
        start_cycle = time.time()
        temp_data_process = get_array_based_on_index(data_process.copy(), overall_keep_list)
        
        if method == "flann":
            flann = FLANN()
            indices, distances = flann.nn(temp_data_process, temp_data_process, 2, algorithm="kmeans")
        elif method == "scipy":
            kd_tree = cKDTree(temp_data_process)
            distances, indices = kd_tree.query(temp_data_process, k=2)
        elif method == "pykdtree":
            kd_tree = KDTree(temp_data_process,leafsize=6)
            distances, indices = kd_tree.query(temp_data_process, k=2)
        elif method == "sklearn":
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree',n_jobs=-1).fit(temp_data_process)
            distances, indices = nbrs.kneighbors(temp_data_process)
        elif method == "annoy":
            annoy = AnnoyIndex(len(temp_data_process[0]),metric='euclidean')
            for i in range(len(temp_data_process)):
                annoy.add_item(i,temp_data_process[i])
            annoy.build(1)
            distances = []
            indices = []
            for i in range(len(temp_data_process)):
                temp_index, temp_dist = annoy.get_nns_by_vector(temp_data_process[i], 2, include_distances=True)
                indices.append([i, temp_index[1]])
                distances.append([0.0, temp_dist[1]])
        elif method == "nmslib":
            index = nmslib.init(method='hnsw', space='l2')
            index.addDataPointBatch(temp_data_process)
            index.createIndex( print_progress=True)

            neighbours = index.knnQueryBatch(temp_data_process, k=2) 
            
            distances = []
            indices = []
            for item in neighbours:
                indices.append(item[0])
                distances.append(item[1])

        else:
            raise NotImplemented
            
    
        remove_index_li = []
        index_li = []
        
        for index, distance in zip(indices, distances):
            index_li.append(index[0])
            if distance[1] <= cutoff:
                #print index
                remove_index_li.append(index[1])
        
        
        temp_num = int(ceil(float(len(remove_index_li))*rate))
                            
        if temp_num == 0:
            keep_going = False
        remove_index_li = random_subsampling(remove_index_li,temp_num)
        
        
        temp_keep_list = remove_list_from_list(index_li, remove_index_li)
        overall_keep_list = [overall_keep_list[i] for i in temp_keep_list ]
        if verbose:
            print('end cycle. length: {}\t time:{}'.format(len(overall_keep_list), time.time()-start_cycle))
       
    print('end subsampling. length: {}\t time:{}'.format(len(overall_keep_list), time.time()-start))
    return overall_keep_list

    
def subsampling_system(data, list_desc = [], standard_scale = True, cutoff_sig = 0.05, rate = 0.3, \
                       method = "pykdtree", verbose = True):
    
    '''
    list_desc: list of dimensions/descriptors used in subsampling
    
    '''
    
    if len(list_desc) == 0:
        data_process = data
    else:
        data_process = get_data_process(data, list_desc)
    
    overall_keep_list = get_subsampling_index2(data_process, standard_scale = standard_scale, \
                                               cutoff_sig = cutoff_sig, method = method, verbose = verbose)
    sampling_result = [data[i] for i in overall_keep_list]
    return sampling_result    



def subsampling_system_with_PCA(data, list_desc = [], standard_scale = True, cutoff_sig = 0.05, rate = 0.3, \
                                start_trial_component = 10, max_component = 30, target_variance = 0.999999, \
                                method = "pykdtree", verbose = True):
    if len(list_desc) == 0:
        data_process = data
    else:
        data_process = get_data_process(data, list_desc)
    
    print('start trial PCA')
    start = time.time()
    pca = PCA( svd_solver = "randomized" )
    data_pca = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    trial_component = start_trial_component-1
    keep_going = True
    while keep_going:
        trial_component +=1
        sum_explained_variance = sum(explained_variance_ratio[:trial_component])
        
        if verbose:
            print("trial components: {} \t explained variance: {}"\
                  .format(trial_component, sum_explained_variance))
        
        if sum_explained_variance > target_variance:
            keep_going = False
               
        if trial_component > max_component:
            keep_going = False
            print("stopped PCA at {} components, total explained variance: {}"\
                  .format(trial_component, sum_explained_variance))

        if trial_component >= len(data_process[0]):
            keep_going = False
            pca_result = data_process
        
        
    print('end trial PCA, number of PC kept: {} \t took {} s'.format(trial_component,str(time.time()-start)))
    
    overall_keep_list = get_subsampling_index2(data_pca[:,:trial_component], standard_scale = standard_scale, \
                                               cutoff_sig = cutoff_sig,rate = rate, method = method, \
                                               verbose = verbose)
    
    sampling_result = [data[i] for i in overall_keep_list]
    return sampling_result 


def subsampling_system_batch(data, list_desc = [], batch_size = 1000000, recursive_level = 2, \
                             standard_scale = True, cutoff_sig = 0.05, rate = 0.3, method = "pykdtree", \
                             verbose = True):
    
    '''
    list_desc: list of dimensions/descriptors used in subsampling
    
    '''
    print("at recursive level {}, length {}".format(recursive_level,len(data)))
    
    sampling_result = []
    for data_subgroup in chunker(data, batch_size):
        temp_sampling_result = subsampling_system(data_subgroup, list_desc = [], standard_scale = standard_scale, \
                                                  cutoff_sig = cutoff_sig, rate = rate, method = method, \
                                                  verbose = verbose)
        sampling_result += temp_sampling_result
    
    print(np.array(sampling_result).shape)
    
    if recursive_level == 1:
        print("at recursive level 1, length {}, Overall subsample".format(recursive_level,len(sampling_result)))
        sampling_result = subsampling_system(sampling_result, list_desc = [], standard_scale = standard_scale, \
                          cutoff_sig = cutoff_sig, rate = rate, method = method, \
                          verbose = verbose)
        print(sampling_result)
        print("returning")
        return sampling_result
        print("after returning")

        
    else:
        print("end recursive level {}, length {}, Continue".format(recursive_level,len(sampling_result )))
        subsampling_system_batch(sampling_result, list_desc = [], batch_size = batch_size, \
                                 recursive_level = recursive_level-1, \
                                 standard_scale = standard_scale, cutoff_sig = cutoff_sig, \
                                 rate = rate, method = method, verbose = verbose)
        

def subsampling_system_with_PCA_batch(data, list_desc = [], batch_size = 1000000, recursive_level = 2, \
                             standard_scale = True, cutoff_sig = 0.05, rate = 0.3, method = "pykdtree", \
                             verbose = True):
    
    '''
    list_desc: list of dimensions/descriptors used in subsampling
    
    '''
    print("at recursive level {}, length {}".format(recursive_level,len(data)))
    
    
    sampling_result = []
    for data_subgroup in chunker(data, batch_size):
        temp_sampling_result = subsampling_system_with_PCA(data_subgroup, list_desc = [], standard_scale = standard_scale, \
                                                  cutoff_sig = cutoff_sig, rate = rate, method = method, \
                                                  verbose = verbose)
        sampling_result += temp_sampling_result
    
    if recursive_level == 1:
        print("at recursive level 1, length {}, Overall subsample".format(recursive_level,len(sampling_result)))
        sampling_result = subsampling_system_with_PCA(sampling_result, list_desc = [], standard_scale = standard_scale, \
                          cutoff_sig = cutoff_sig, rate = rate, method = method, \
                          verbose = verbose)

        return sampling_result
        
    else:
        print("end recursive level {}, length {}, Continue".format(recursive_level,len(sampling_result )))
        subsampling_system_with_PCA_batch(sampling_result, list_desc = [], batch_size = batch_size, \
                                 recursive_level = recursive_level-1, \
                                 standard_scale = standard_scale, cutoff_sig = cutoff_sig, \
                                 rate = rate, method = method, verbose = verbose)



def random_subsampling(li,num):
    '''
    return chosen number of random entries from a list
    '''
    if num == 0:
        return []
    elif len(li) > num:
        # generate random list of indecies
        index_list = random.sample(range(0, len(li)-1),num)
        return [li[i] for i in index_list]

    else:
        print('no sampling')
        return li
        
        
import matplotlib
import matplotlib.pyplot as plt
def plot_stuff(dataframe,filename):
   sns.set(font_scale = 2)
   with sns.axes_style('white'):
       #g = sns.jointplot('x', 'y', data=dataframe,size=10,s=2,stat_func=None,marker='o', space = 0)
       g = sns.JointGrid(x="x", y="y",data=dataframe,space=0)
       g = g.plot_joint(plt.scatter, color = "b", s=30)

       g.set_axis_labels(fontsize=50)

       count = lambda a, b: len(a)
       g.annotate(count, template="{val:.0f}",stat="Count",loc="upper right", fontsize = 20)

       _ = g.ax_marg_x.hist(dataframe["x"], color = "b", alpha = 0.6, bins = np.arange(-0.5, 0.5, 0.02))
       _ = g.ax_marg_y.hist(dataframe["y"], color = "b", alpha = 0.6, orientation = "horizontal",bins = np.arange(0, 200, 4))
       #figure = temp_plot.get_figure()
       plt.show()
       #plt.savefig(filename)
       
import random
import seaborn as sns
import pandas as pd
n = 1000000
x1_mu, x1_sigma = 0, 0.1
x2_mu, x2_sigma = 100, 20

x1 = np.random.normal(x1_mu, x1_sigma, n)
x2 = np.random.normal(x2_mu, x2_sigma, n)

df = pd.DataFrame({"x":x1, "y":x2,"group":np.repeat("original",n)})
plot_stuff(df,"test.png")

import time
temp = np.column_stack((x1,x2))
start = time.time()
temp_subsampled = subsampling_system_batch(temp,list_desc = [], batch_size = 100000, recursive_level = 2, \
                                                      cutoff_sig=0.05,rate=0.2, method = "pykdtree",verbose = False)

print(temp_subsampled.shape)
#print(subsampling_system_batch(temp,list_desc = [], batch_size = 100000, recursive_level = 2, \
#                                     cutoff_sig=0.05,rate=0.2, method = "pykdtree",verbose = False))
print("after")
