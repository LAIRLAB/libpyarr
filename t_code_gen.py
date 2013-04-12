#! /usr/bin/env python

from code_gen import *

print vec_decl('double')
print vec_decl('double_vec')
print vec_decl('uint', 'unsigned int')
print vec_decl('ulong', 'unsigned long')

print cls_decl('rh::RegionData', [], ['m_features', 'm_temp_labels']).gen_boost()

print cls_decl('VRandomForest::VTreeNode', 
               [['get_left', 'ext', 'reo'],
                ['get_right', 'ext', 'reo'],
                ['refill_avg_outputs', 'ext']], 

               ['is_leaf', 'count', 'dim', 'avg_output', 
                ['thresh', 'ro']]).gen_boost()
print vec_decl('VTreeNode')
                                            
vr= cls_decl('VRandomForest', 
               [],
               ['m_nbr_trees', 
                'm_max_depth', 
                'm_min_node_size', 
                'm_rdm_dim_percent', 
                'm_rdm_sample_percent', 
                'm_seeds', 
                'm_trees'])
print vr.gen()
print vr.gen_boost()
print vec_decl('VRandomForest')

print cls_decl('VBoostedMaxEnt', 
               [['get_vec_vrandomforest', 'ext'], 
                ['wrap_train', 'ext'], 
                ['wrap_predict', 'ext'], 
                'save', 
                'load'], 
               ['m_step_sizes', 
                'm_dim'], 
               init_args=['double', 'double', 'double', 
                          'int', 'VRandomForest']).gen_boost()


print cls_decl('tBox', 
               [],
               ['type', 'x1', 'y1', 'x2', 'y2', 'alpha'], 
               init_args=['string', 
                          'double', 
                          'double', 
                          'double', 
                          'double', 
                          'double']).gen_boost()

print cls_decl('tGroundtruth', 
               [], 
               ['box', 'truncation', 'occlusion']).gen_boost()

print cls_decl('tDetection', 
               [],
               ['box', 'thresh']).gen_boost()
               
print vec_decl('tGroundtruth')
print vec_decl('tDetection')
print vec_decl('tGroundtruth_vec')
print vec_decl('tDetection_vec')

convs = [imarr_converter('int', 'NPY_INT32'),
         imarr_converter('unsigned int', 'NPY_UINT32'),
         imarr_converter('double', 'NPY_FLOAT64'),
         imarr_converter('float', 'NPY_FLOAT32'),
         imarr_converter('unsigned char', 'NPY_UINT8')]

for c in convs:
    print c.gen()
print 
for c in convs:
    print c.gen_reg()
