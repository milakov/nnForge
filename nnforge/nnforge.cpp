#include "nnforge.h"

#include "layer_factory.h"

namespace nnforge
{
	void nnforge::init()
	{
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new convolution_layer(std::vector<unsigned int>(1, 1), 1, 1)));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new hyperbolic_tangent_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new average_subsampling_layer(std::vector<unsigned int>(1, 1))));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new max_subsampling_layer(std::vector<unsigned int>(1, 1))));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new absolute_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new local_contrast_subtractive_layer(std::vector<unsigned int>(1, 1), std::vector<unsigned int>(1, 0), 1)));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new rgb_to_yuv_convert_layer(std::vector<color_feature_map_config>(1, color_feature_map_config(0, 1, 2)))));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new rectified_linear_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new soft_rectified_linear_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new softmax_layer()));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new maxout_layer(2)));
		single_layer_factory::get_mutable_instance().register_layer(layer_smart_ptr(new sigmoid_layer()));
	}
}
