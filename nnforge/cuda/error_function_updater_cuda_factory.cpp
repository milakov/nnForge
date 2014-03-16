/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "error_function_updater_cuda_factory.h"

#include "../neural_network_exception.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		bool error_function_updater_cuda_factory::register_error_function_updater_cuda(error_function_updater_cuda_smart_ptr sample_error_function_updater_cuda)
		{
			return sample_error_function_updater_cuda_map.insert(sample_map::value_type(sample_error_function_updater_cuda->get_uuid(), sample_error_function_updater_cuda)).second;
		}

		bool error_function_updater_cuda_factory::unregister_error_function_updater_cuda(const boost::uuids::uuid& error_function_guid)
		{
			return sample_error_function_updater_cuda_map.erase(error_function_guid) == 1;
		}

		const_error_function_updater_cuda_smart_ptr error_function_updater_cuda_factory::get_error_function_updater_cuda(const boost::uuids::uuid& error_function_guid) const
		{
			sample_map::const_iterator i = sample_error_function_updater_cuda_map.find(error_function_guid);

			if (i == sample_error_function_updater_cuda_map.end())
				throw neural_network_exception((boost::format("No CUDA error function updater is registered with id %1%") % error_function_guid).str());

			return i->second;
		}
	}
}
