#include <boost_common.h>
#include <boilerplate.cpp>
#include <autogen_converters.cpp>
#include <to_python_converters.cc>

using std::string;
using std::vector;

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

void VRandomForest__wrap_train(VRandomForest *inst, 
                                 pyarr<double> X, 
                                 pyarr<double> Y)
{
    if (X.dims[0] != Y.dims[0]) {
        printf("OH NO! X.dims[0] = %ld, Y.dims[0] = %ld\n", 
               X.dims[0], Y.dims[0]);
        return;
    }
    vector<const vector<double>*> Xv, Yv;
    for (int i=0; i<X.dims[0]; i++) {
        vector<double>* Xtmp = new vector<double>(X.dims[1]);
        vector<double>* Ytmp = new vector<double>(Y.dims[1]);
        for (int j=0; j<X.dims[1]; j++) {
            (*Xtmp)[j] = X[ind(i,j)];
        }
        for (int j=0; j<Y.dims[1]; j++) {
            (*Ytmp)[j] = Y[ind(i,j)];
        }
        Xv.push_back(const_cast<const vector<double>*>(Xtmp));
        Yv.push_back(const_cast<const vector<double>*>(Ytmp));
    }
    vector<double> weights(X.dims[0], 1.0), feature_costs;
    vector<size_t> usable_features, required_features;
    
    inst->train(X.dims[1], Y.dims[1], 
                Xv, Yv, weights, usable_features, feature_costs, required_features);
    for (int i=0; i<X.dims[0]; i++) {
        delete Xv[i];
        delete Yv[i];
    }
}

pyarr<double> VRandomForest__wrap_predict(VRandomForest *inst, 
                                            pyarr<double> X)
{
    long int argh[] = {inst->getDimTarget()};
    pyarr<double> ret(1, argh);
    
    vector<double> xv(X.dims[0]), yv(ret.dims[0]);
    for (int i=0; i<X.dims[0]; i++) {
        xv[i] = X[ind(i)];
    }
    inst->predict(xv, yv);
    for (int i=0; i<ret.dims[0]; i++){
        ret[ind(i)] = yv[i];
    }
    return ret;
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
    import_array();
    if (boosted_common) return; 
    boosted_common = true;
    

    // long int dims[] = {10, 10, 10};
    // pyarr<double> d(3, dims);
    // pyarr<float> f(3, dims);
    // pyarr<int> i(3, dims);
    // pyarr<long int> l(3, dims);
    // pyarr<unsigned char> uc(3, dims);
    // pyarr<unsigned int> ui(3, dims);
    // pyarr<char> c(3, dims);

    // printf("d %d f %d i %d l %d uc %d ui %d c %d\n", 
    //        d.ao->descr->type_num, 
    //        f.ao->descr->type_num, 
    //        i.ao->descr->type_num, 
    //        l.ao->descr->type_num, 
    //        uc.ao->descr->type_num, 
    //        ui.ao->descr->type_num, 
    //        c.ao->descr->type_num);


    to_python_converter<MatrixMap, MatrixMap_to_numpy_str>();
    MatrixMap_from_numpy_str();

    to_python_converter<d2d::SegmentMap, SegmentMap_to_numpy_str>();
    SegmentMap_from_numpy_str();

    to_python_converter<f2d::Pixel2DData, Pixel2DData_to_numpy_str>();
    Pixel2DData_from_numpy_str();

    to_python_converter<LRgbImage, LRgbImage_to_numpy_str>();
    LRgbImage_from_numpy_str();

    to_python_converter<vector<double>, vec_to_numpy_str>();
    vec_from_numpy_str();

    register_autogen_converters();
    register_common_converters();

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

    class_<vector<string> >("string_vector")
	.def(vector_indexing_suite<vector<string > >() )
	;


    
}

static bool boosted_ml = false;

void boost_ml() 
{
    if (0 && boosted_ml) return; 
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
        .def("save", &VRandomForest::save)
        .def("load", &VRandomForest::load)
        .def("train", VRandomForest__wrap_train)
        .def("predict", VRandomForest__wrap_predict)
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

BOOST_PYTHON_MODULE(libboost_common) {
    PyEval_InitThreads();
    import_array();
    ilInit();
    boost_common();
}
