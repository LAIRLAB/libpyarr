#pragma once

#include <boost_common.h>
#include <pyarr.h>

#include <stdexcept>
using boost::python::object;

// create dummy fill_final for every level except the lowest level, at which we call the real one
template<typename R, typename T>
    void pyarr_fill_final(vector<R> dummy1, pyarr<T> dummy2, vector<size_t> dummy3,
			  typename boost::enable_if<boost::is_class<R>, int >::type template_dummy = 0) {}

// fill a vector (last dimension of the pyarr) with T data from the pyarr<T>, 
// disable creating functions if we're not at the lowest level
template<typename R, typename T>
    void pyarr_fill_final(vector<T> &v, pyarr<T> arr, vector<size_t> pre_idx,
			  typename boost::disable_if<boost::is_class<R>, int >::type template_dummy = 0)
{
    for (size_t x = 0; x < arr.dims.back(); x++)
	{
	    vector<size_t> full_idx = pre_idx;
	    full_idx.push_back(x);
	    v.push_back(arr[ind(full_idx)]);
	}
}

// create dummy fill vectors for every level but the lowest level
template<typename R, typename T>
    void pyarr_fill_vector(R dummy, int dummy2, pyarr<T> dummy3, vector<size_t> pre_idx,
			   typename boost::disable_if<boost::is_class<R>, int >::type template_dummy = 0) {}


// recursively instatiate and create a vector tensor from a pyarr (helper function). enable recursive
// template declarations if we're not at the lowest level
template<typename R, typename T>
    void pyarr_fill_vector(R &v, int nd, pyarr<T> arr, vector<size_t> pre_idx,
			   typename boost::enable_if<boost::is_class<R>, int >::type template_dummy = 0)
{
    if (nd > 1)
	{
	    for (size_t x = 0; x < arr.dims[pre_idx.size()]; x++)
		{
		    typename R::value_type new_v;
	    
		    vector<size_t> new_pre_idx = pre_idx;

		    new_pre_idx.push_back(x);
	    
		    pyarr_fill_vector<typename R::value_type, T>(new_v,
								 nd - 1,
								 arr,
								 new_pre_idx);
		    v.push_back(new_v);
		}
	}
    else
	{
	pyarr_fill_final<typename R::value_type, T>(v, arr, pre_idx);
	}
}

template <class A, class B>
struct CheckTypes
{
    static const bool value = false;
};

template <class A>
struct CheckTypes<A, A>
{
    static const bool value = true;
};


// convert a pyarr to a vector tensor
template<typename R, typename T> R pyarr_to_v_tensor(pyarr<T> arr)
{
    R final_data;
    vector<size_t> pre_idx;

    pyarr_fill_vector<R, T>(final_data, arr.dims.size(), arr, pre_idx);
    return final_data;
}

