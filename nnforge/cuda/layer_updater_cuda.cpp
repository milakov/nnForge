/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "layer_updater_cuda.h"

#include "util_cuda.h"
#include "neural_network_cuda_exception.h"

namespace nnforge
{
	namespace cuda
	{
		void layer_updater_cuda::configure(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			layer::const_ptr layer_schema,
			cuda_running_configuration::const_ptr cuda_config,
			const std::set<layer_action>& actions)
		{
			this->layer_schema = layer_schema;
			this->input_configuration_specific_list = input_configuration_specific_list;
			this->output_configuration_specific = output_configuration_specific;
			this->cuda_config = cuda_config;
			this->actions = actions;

			input_elem_count_per_entry_list.resize(input_configuration_specific_list.size());
			input_elem_count_per_feature_map_list.resize(input_configuration_specific_list.size());
			for(int i = 0; i < input_configuration_specific_list.size(); ++i)
			{
				input_elem_count_per_entry_list[i] = input_configuration_specific_list[i].get_neuron_count();
				input_elem_count_per_feature_map_list[i] = input_configuration_specific_list[i].get_neuron_count_per_feature_map();
			}

			output_elem_count_per_entry = output_configuration_specific.get_neuron_count();
			output_elem_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();

			updater_configured();
		}

		void layer_updater_cuda::updater_configured()
		{
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_updater_cuda::get_persistent_working_data() const
		{
			return std::vector<cuda_linear_buffer_device::const_ptr>();
		}

		std::vector<unsigned int> layer_updater_cuda::get_linear_addressing_through_texture_per_entry() const
		{
			return std::vector<unsigned int>();
		}

		std::vector<cuda_linear_buffer_device::ptr> layer_updater_cuda::get_data(layer_data::const_ptr host_data) const
		{
			std::vector<cuda_linear_buffer_device::ptr> res;

			for(std::vector<std::vector<float> >::const_iterator it = host_data->begin(); it != host_data->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(float);
				cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		std::vector<cuda_linear_buffer_device::const_ptr> layer_updater_cuda::set_get_data_custom(layer_data_custom::const_ptr host_data_custom)
		{
			notify_data_custom(host_data_custom);

			std::vector<cuda_linear_buffer_device::const_ptr> res;

			for(std::vector<std::vector<int> >::const_iterator it = host_data_custom->begin(); it != host_data_custom->end(); ++it)
			{
				size_t buffer_size = it->size() * sizeof(int);
				cuda_linear_buffer_device::ptr new_buf(new cuda_linear_buffer_device(buffer_size));
				cuda_safe_call(cudaMemcpy(*new_buf, &(*it->begin()), buffer_size, cudaMemcpyHostToDevice));
				res.push_back(new_buf);
			}

			return res;
		}

		void layer_updater_cuda::get_data_from_device(const std::vector<cuda_linear_buffer_device::ptr>& device_data, layer_data::ptr host_data) const
		{
			unsigned int part_id = 0;
			for(layer_data::iterator it = host_data->begin(); it != host_data->end(); ++it, ++part_id)
			{
				cuda_linear_buffer_device::const_ptr src = device_data[part_id];
				cuda_safe_call(cudaMemcpy(&(*it->begin()), *src, it->size() * sizeof(float), cudaMemcpyDeviceToHost));
			}
		}

		void layer_updater_cuda::notify_data_custom(layer_data_custom::const_ptr host_data_custom)
		{
		}

		std::pair<size_t, bool> layer_updater_cuda::get_temporary_working_fixed_buffer_size(const layer_action& action) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_working_fixed_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return std::make_pair(0, false);
		}

		size_t layer_updater_cuda::get_temporary_working_per_entry_buffer_size(const layer_action& action) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_working_per_entry_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return 0;
		}

		size_t layer_updater_cuda::get_temporary_fixed_buffer_size() const
		{
			if (actions.find(layer_action(layer_action::forward)) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_fixed_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % layer_action(layer_action::forward).str()).str());

			return 0;
		}

		size_t layer_updater_cuda::get_temporary_per_entry_buffer_size() const
		{
			if (actions.find(layer_action(layer_action::forward)) == actions.end())
				throw neural_network_exception((boost::format("get_temporary_per_entry_buffer_size called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % layer_action(layer_action::forward).str()).str());

			return 0;
		}

		int layer_updater_cuda::get_input_index_layer_can_write(const layer_action& action) const
		{
			if (actions.find(action) == actions.end())
				throw neural_network_exception((boost::format("get_input_index_layer_can_write called for layer %1% for action %2% while it is not configured to run such an action") % layer_schema->instance_name % action.str()).str());

			return -1;
		}

		bool layer_updater_cuda::is_backward_data_dependent_on_input_buffer(unsigned int action_input_index, unsigned int data_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_input_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return true;
		}

		bool layer_updater_cuda::is_backward_data_dependent_on_output_buffer(unsigned int action_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_output_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return true;
		}

		bool layer_updater_cuda::is_backward_data_dependent_on_temporary_fixed_buffer(unsigned int action_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_temporary_fixed_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return (get_temporary_fixed_buffer_size() != 0);
		}

		bool layer_updater_cuda::is_backward_data_dependent_on_temporary_per_entry_buffer(unsigned int action_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_data, action_input_index)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_dependent_on_temporary_per_entry_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data, action_input_index).str()).str());

			return (get_temporary_per_entry_buffer_size() != 0);
		}

		bool layer_updater_cuda::is_backward_data_and_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_data_and_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_and_weights_dependent_on_input_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data_and_weights).str()).str());

			return true;
		}

		bool layer_updater_cuda::is_backward_data_and_weights_dependent_on_output_buffer() const
		{
			if (actions.find(layer_action(layer_action::backward_data_and_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_and_weights_dependent_on_output_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data_and_weights).str()).str());

			return true;
		}

		bool layer_updater_cuda::is_backward_data_and_weights_dependent_on_temporary_fixed_buffer() const
		{
			if (actions.find(layer_action(layer_action::backward_data_and_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_and_weights_dependent_on_temporary_fixed_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data_and_weights).str()).str());

			return (get_temporary_fixed_buffer_size() != 0);
		}

		bool layer_updater_cuda::is_backward_data_and_weights_dependent_on_temporary_per_entry_buffer() const
		{
			if (actions.find(layer_action(layer_action::backward_data_and_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_data_and_weights_dependent_on_temporary_per_entry_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_data_and_weights).str()).str());

			return (get_temporary_per_entry_buffer_size() != 0);
		}

		bool layer_updater_cuda::is_backward_weights_dependent_on_input_buffer(unsigned int data_input_index) const
		{
			if (actions.find(layer_action(layer_action::backward_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_weights_dependent_on_input_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_weights).str()).str());

			return true;
		}

		bool layer_updater_cuda::is_backward_weights_dependent_on_temporary_fixed_buffer() const
		{
			if (actions.find(layer_action(layer_action::backward_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_weights_dependent_on_temporary_fixed_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_weights).str()).str());

			return (get_temporary_fixed_buffer_size() != 0);
		}

		bool layer_updater_cuda::is_backward_weights_dependent_on_temporary_per_entry_buffer() const
		{
			if (actions.find(layer_action(layer_action::backward_weights)) == actions.end())
				throw neural_network_exception((boost::format("is_backward_weights_dependent_on_temporary_per_entry_buffer called for layer %1% while it is not configured to run action %2%") % layer_schema->instance_name % layer_action(layer_action::backward_weights).str()).str());

			return (get_temporary_per_entry_buffer_size() != 0);
		}

		void layer_updater_cuda::enqueue_backward_data_propagation(
			cudaStream_t stream_id,
			unsigned int input_index,
			cuda_linear_buffer_device::ptr input_errors_buffer,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			throw neural_network_exception((boost::format("enqueue_backward_data_propagation is not implemented for layer %1%") % layer_schema->instance_name).str());
		}

		void layer_updater_cuda::enqueue_backward_weights_propagation(
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& gradient,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			unsigned int entry_count)
		{
			throw neural_network_exception((boost::format("enqueue_backward_weights_propagation is not implemented for layer %1%") % layer_schema->instance_name).str());
		}

		void layer_updater_cuda::enqueue_backward_data_and_weights_propagation(
			cudaStream_t stream_id,
			const std::vector<cuda_linear_buffer_device::ptr> input_errors_buffers,
			cuda_linear_buffer_device::const_ptr output_errors_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
			const std::vector<cuda_linear_buffer_device::ptr>& gradient,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data,
			const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
			const std::vector<cuda_linear_buffer_device::const_ptr>& input_neurons_buffers,
			cuda_linear_buffer_device::const_ptr output_neurons_buffer,
			const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
			cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
			cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
			cuda_linear_buffer_device::const_ptr temporary_fixed_buffer,
			cuda_linear_buffer_device::const_ptr temporary_per_entry_buffer,
			bool add_update_to_destination,
			unsigned int entry_count)
		{
			throw neural_network_exception((boost::format("enqueue_backward_data_and_weights_propagation is not implemented for layer %1%") % layer_schema->instance_name).str());
		}
	}
}
