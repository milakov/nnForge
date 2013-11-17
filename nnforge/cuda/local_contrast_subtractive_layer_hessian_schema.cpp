/*
 *  Copyright 2011-2013 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "local_contrast_subtractive_layer_hessian_schema.h"

#include "../local_contrast_subtractive_layer.h"
#include "../neural_network_exception.h"
#include "local_contrast_subtractive_2d_layer_hessian_cuda.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		local_contrast_subtractive_layer_hessian_schema::local_contrast_subtractive_layer_hessian_schema()
		{
		}

		local_contrast_subtractive_layer_hessian_schema::~local_contrast_subtractive_layer_hessian_schema()
		{
		}

		std::tr1::shared_ptr<layer_hessian_schema> local_contrast_subtractive_layer_hessian_schema::create_specific() const
		{
			return layer_hessian_schema_smart_ptr(new local_contrast_subtractive_layer_hessian_schema());
		}

		const boost::uuids::uuid& local_contrast_subtractive_layer_hessian_schema::get_uuid() const
		{
			return local_contrast_subtractive_layer::layer_guid;
		}

		layer_hessian_cuda_smart_ptr local_contrast_subtractive_layer_hessian_schema::create_hessian_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			layer_hessian_cuda_smart_ptr res;

			switch (output_configuration_specific.dimension_sizes.size())
			{
			case 2:
				res = layer_hessian_cuda_smart_ptr(new local_contrast_subtractive_2d_layer_hessian_cuda());
				break;
			default:
				throw neural_network_exception((boost::format("No CUDA hessian for the local contrast subtractive layer of %1% dimensions") % output_configuration_specific.dimension_sizes.size()).str());
				break;
			}

			return res;
		}

		std::vector<const_cuda_linear_buffer_device_smart_ptr> local_contrast_subtractive_layer_hessian_schema::get_schema_buffers() const
		{
			std::vector<const_cuda_linear_buffer_device_smart_ptr> res;

			std::tr1::shared_ptr<const local_contrast_subtractive_layer> layer_derived = std::tr1::dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);

			res.push_back(
				cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
					&(*layer_derived->feature_maps_affected.begin()),
					layer_derived->feature_maps_affected.size() * sizeof(unsigned int)))
				);

			for(std::vector<std::vector<float> >::const_iterator it = layer_derived->window_weights_list.begin(); it != layer_derived->window_weights_list.end(); ++it)
			{
				const std::vector<float>& current_weights = *it;
				res.push_back(
					cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*current_weights.begin()),
						current_weights.size() * sizeof(float)))
					);
			}

			for(std::vector<std::vector<float> >::const_iterator it = layer_derived->window_weights_list.begin(); it != layer_derived->window_weights_list.end(); ++it)
			{
				const std::vector<float>& current_weights = *it;
				std::vector<float> current_weights_squared;
				for(std::vector<float>::const_iterator it2 = current_weights.begin(); it2 != current_weights.end(); ++it2)
					current_weights_squared.push_back(*it2 * *it2);
				res.push_back(
					cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*current_weights_squared.begin()),
						current_weights_squared.size() * sizeof(float)))
					);
			}

			if (!layer_derived->feature_maps_unaffected.empty())
			{
				res.push_back(
					cuda_linear_buffer_device_smart_ptr(new cuda_linear_buffer_device(
						&(*layer_derived->feature_maps_unaffected.begin()),
						layer_derived->feature_maps_unaffected.size() * sizeof(unsigned int)))
					);
			}

			return res;
		}
	}
}
