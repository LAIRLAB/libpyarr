#include <boost_common.h>
#include <boilerplate.cpp>
#include <autogen_converters.cpp>

/* at some point I gotta figure out how to make readonly properties of pointer 
 * fields without this crap 
 */
VRandomForest::VTreeNode* VTreeNode__get_left(VRandomForest::VTreeNode* inst)
{
    return inst->left;
}
VRandomForest::VTreeNode* VTreeNode__get_right(VRandomForest::VTreeNode* inst)
{
    return inst->right;
}
bool VBoostedMaxEnt__wrap_train(VBoostedMaxEnt* inst, 
                                PyObject* X_train, 
                                PyObject* Y_train)
{
    PyArrayObject *xao = (PyArrayObject*)X_train;
    PyArrayObject *yao = (PyArrayObject*)Y_train;

    if (xao->dimensions[0] != yao->dimensions[0]) {
      printf("oh no, X train has %zu entries but Y train has %zu\n",
             xao->dimensions[0],
             yao->dimensions[0]);
    }

    vector<const vector<double>*> Xvec;
    vector<const vector<double>*> Yvec;

    for (int i=0; i<xao->dimensions[0]; i++) {
      vector<double> *tmp = new vector<double>(((double*)xao->data) + i*xao->dimensions[1],
                                               ((double*)xao->data) + (i+1)*xao->dimensions[1]);
      Xvec.push_back(const_cast<const vector<double>*>(tmp));

      tmp = new vector<double>(&(((double*)yao->data)[i*yao->dimensions[1]]), 
                               &(((double*)yao->data)[(i+1)*yao->dimensions[1]]));
      double total_prob = 0.0;
      for (int j=0; j<yao->dimensions[1]; j++) {
          total_prob += (*tmp)[j];
      }
      if (total_prob >= 1.001) {
          printf("OH NO!!! total_prob = %f, tmp[0] = %f, tmp[1] = %f, i=%d\n", 
                 total_prob, (*tmp)[0], (*tmp)[1], i);
          printf("actual data at (%d,%d) and (%d,%d) was %f and %f, absolute coords %zu and %zu\n", 
                 i,0, i,1, ((double*)yao->data)[i*yao->dimensions[1] + 0], 
                 ((double*)yao->data)[i*yao->dimensions[1] + 1], 
                 i*yao->dimensions[1], i*yao->dimensions[1] + 1);
          return true;
      }
      Yvec.push_back(const_cast<const vector<double>*>(tmp));
    }
    vector<double> w_fold(Xvec.size(), 1.0);
    inst->train(Xvec, Yvec, w_fold, NULL, NULL);

    for (int i=0; i<xao->dimensions[0]; i++) {
      delete Xvec[i];
      delete Yvec[i];
    }
    return false;
}

PyObject* VBoostedMaxEnt__wrap_predict(VBoostedMaxEnt* inst, 
                                       vector<double> query)
{
    vector<double> ret(2);
    inst->predict(query, ret);
    return vec_to_numpy(ret);
}



vector<VRandomForest*> VBoostedMaxEnt__get_vec_vrandomforest(VBoostedMaxEnt* inst)
{
  vector<VRandomForest*> ret;
  foreach (it, inst->m_vregressors) {
    /* C++ is beautiful: 
     * - the innermost * dereferences the iterator;
     * - the next * dereferences the shared ptr; 
     * - the & gets the address of the actual object, because boost can't deal with std
     *     tr1 shared ptrs
     * - it's officially a vector of VRegressors, but we happen to know they're really 
     *     VRandomForests, so cast. 
     */
    ret.push_back((VRandomForest*)&(*(*it)));
  }
  return ret;
}
void VTreeNode__refill_avg_outputs(VRandomForest::VTreeNode *inst) 
{
    if (inst->avg_output.size() > 0) 
        return;

    if (inst->left->avg_output.size() == 0) 
        VTreeNode__refill_avg_outputs(inst->left);
    if (inst->right->avg_output.size() == 0) 
        VTreeNode__refill_avg_outputs(inst->right);
    
    for (int i=0; i<inst->left->avg_output.size(); i++) {
        inst->avg_output.push_back((inst->left->avg_output[i] +
                                    inst->right->avg_output[i]) / 2.0);
    }
}

void set_logger_verbosity(string verbosity)
{
    misc_util::Logger::status_t s;
    if (verbosity == "debug") s = misc_util::Logger::DEBUG;
    else if (verbosity == "info") s = misc_util::Logger::INFO;
    else if (verbosity == "warning") s = misc_util::Logger::WARNING;
    else if (verbosity == "error") s = misc_util::Logger::ERROR;
    else s = misc_util::Logger::INFO;

    misc_util::Logger::setVerbosity(s);
}

static bool boosted_common = false;

void boost_common() 
{
    if (boosted_common) return; 
    boosted_common = true;

    to_python_converter<MatrixMap, MatrixMap_to_numpy_str>();
    MatrixMap_from_numpy_str();

    to_python_converter<d2d::SegmentMap, SegmentMap_to_numpy_str>();
    SegmentMap_from_numpy_str();

    to_python_converter<mxArray, mxArray_to_numpy_str>();
    mxArray_from_numpy_str();

    to_python_converter<f2d::Pixel2DData, Pixel2DData_to_numpy_str>();
    Pixel2DData_from_numpy_str();

    to_python_converter<LRgbImage, LRgbImage_to_numpy_str>();
    LRgbImage_from_numpy_str();

    to_python_converter<vector<double>, vec_to_numpy_str>();
    vec_from_numpy_str();

    register_autogen_converters();

    class_<vector<double> >("double_vec")
        .def(vector_indexing_suite<vector<double> >())
        ;
    class_<vector<unsigned int> >("uint_vec")
        .def(vector_indexing_suite<vector<unsigned int> >())
        ;
    class_<vector<unsigned long> >("ulong_vec")
        .def(vector_indexing_suite<vector<unsigned long> >())
        ;
    class_<vector<vector<double> > >("double_vec_vec")
        .def(vector_indexing_suite<vector<vector<double> > >())
        ;
    class_<vector<vector<vector<double> > > >("double_vec_vec_vec")
        .def(vector_indexing_suite<vector<vector<vector<double> > > >())
        ;
    class_<std::pair<unsigned int, double> >("uint_double_pair")
        .def_readwrite("first", &std::pair<unsigned int, double>::first)
        .def_readwrite("second", &std::pair<unsigned int, double>::second)
        ;
    class_<vector<std::pair<unsigned int, double> > >("uint_double_pair_vec")
        .def(vector_indexing_suite<vector<std::pair<unsigned int, double> > >())
        ;
    class_<vector<vector<std::pair<unsigned int, double> > > >("uint_double_pair_vec_vec")
        .def(vector_indexing_suite<vector<vector<std::pair<unsigned int, double> > > >())
        ;
    def("set_logger_verbosity", set_logger_verbosity);
}

static bool boosted_ml = false;

void boost_ml() 
{
    if (boosted_ml) return; 
    boosted_ml = true;

    class_<VRandomForest::VTreeNode>("VTreeNode", init<>())
        .def_readwrite("is_leaf", &VRandomForest::VTreeNode::is_leaf)
        .def_readwrite("count", &VRandomForest::VTreeNode::count)
        .def_readwrite("dim", &VRandomForest::VTreeNode::dim)
        .def_readwrite("avg_output", &VRandomForest::VTreeNode::avg_output)
        .def("get_left", VTreeNode__get_left, return_value_policy<reference_existing_object>())
        .def("get_right", VTreeNode__get_right, return_value_policy<reference_existing_object>())
        .def("refill_avg_outputs", VTreeNode__refill_avg_outputs)
        .def_readonly("thresh", &VRandomForest::VTreeNode::thresh)
        ;

    class_<vector<VRandomForest::VTreeNode> >("VTreeNode_vec")
        .def(vector_indexing_suite<vector<VRandomForest::VTreeNode> >())
        ;

    class_<VRandomForest>("VRandomForest", init<int, int, int, double, double>())
        .def_readwrite("m_nbr_trees", &VRandomForest::m_nbr_trees)
        .def_readwrite("m_max_depth", &VRandomForest::m_max_depth)
        .def_readwrite("m_min_node_size", &VRandomForest::m_min_node_size)
        .def_readwrite("m_rdm_dim_percent", &VRandomForest::m_rdm_dim_percent)
        .def_readwrite("m_rdm_sample_percent", &VRandomForest::m_rdm_sample_percent)
        .def_readwrite("m_seeds", &VRandomForest::m_seeds)
        .def_readwrite("m_trees", &VRandomForest::m_trees)
        ;
    class_<vector<VRandomForest*> >("VRandomForest_vec")
        .def(vector_indexing_suite<vector<VRandomForest*> >())
        ;
    class_<VBoostedMaxEnt>("VBoostedMaxEnt", init<double, double, double, int, VRandomForest>())
        .def_readwrite("m_step_sizes", &VBoostedMaxEnt::m_step_sizes)
        .def_readwrite("m_dim", &VBoostedMaxEnt::m_dim)
        .def("get_vec_vrandomforest", VBoostedMaxEnt__get_vec_vrandomforest)
        .def("train", VBoostedMaxEnt__wrap_train)
        .def("predict", VBoostedMaxEnt__wrap_predict)
        .def("save", &VBoostedMaxEnt::save)
        .def("load", &VBoostedMaxEnt::load)
        ;
    class_<vector<VBoostedMaxEnt*> >("VBoostedMaxEnt_vec")
        .def(vector_indexing_suite<vector<VBoostedMaxEnt*> >())
        ;
}
