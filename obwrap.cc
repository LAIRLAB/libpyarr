//#define PY_ARRAY_UNIQUE_SYMBOL

#include "OB_CPP_release/OBmain.cpp"

#include "boilerplate.cpp"

//#define PY_ARRAY_UNIQUE_SYMBOL 23847


class ob_runner {
public:
    vector<CModel> models;

    ob_runner(list l) {
        int length = len(l);
        for (int i=0;i<length;i++) {
            models.push_back(CModel(extract<string>(l[i])));
        }
    }

    vector<vector<mxArray*> > get_ob(mxArray* im)
    {
        vector<vector<mxArray*> > responsemap;
        int min_edge_len;
        double ratio;
        vector<float> outputFeature;

        int numLevels = 6;                        //DEFAULT: 6 scale levels
        int numComponents = 2;                //DEFAULT: 2 Components

        min_edge_len = min(im->Dims[0], im->Dims[1]);

        ratio = (float)g_MIN_EDGE_LEN / min_edge_len;                
        mxArray* input_resized = resize(im, ratio);

        //Check if output file for this file already exists:

        printf("got %d models\n", models.size());

        extractOBFeature(input_resized, 
                         models,
                         false,
                         "blah", 
                         outputFeature, 
                         responsemap,
                         numComponents, 
                         numLevels);

        mxFree(&input_resized);

        return responsemap;
    }

    list py_get_ob(PyObject *image)
    {
        list pyret;
        mxArray *mxim = numpy_to_mxarray3d(image);
        vector<vector<mxArray*> > ret = get_ob(mxim);
        mxFree(&mxim);

        /* investigate whether this leaks */ 
        foreach (it, ret) {
            list l;
            foreach(it2, (*it)) {
                l.append(handle<>(mxarray2d_to_numpy(*it2)));
            }
            pyret.append(l);
            //	    ClearMXVec(*it);
        }

        return pyret;
    }
};

BOOST_PYTHON_MODULE(libobwrap)
{
    import_array();
    class_<ob_runner>("ob_runner", init<list>())
        .def("get_ob", &ob_runner::py_get_ob)
        ;
}
