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

#pragma once

#include "error_function_updater_cuda.h"

#include <map>
#include <vector>
#include <boost/uuid/uuid.hpp>
#include <boost/serialization/singleton.hpp>

namespace nnforge
{
	namespace cuda
	{
		class error_function_updater_cuda_factory
		{
		public:
			bool register_error_function_updater_cuda(error_function_updater_cuda_smart_ptr sample_error_function_updater_cuda);

			bool unregister_error_function_updater_cuda(const boost::uuids::uuid& error_function_guid);

			const_error_function_updater_cuda_smart_ptr get_error_function_updater_cuda(const boost::uuids::uuid& error_function_guid) const;

		private:
			typedef std::map<boost::uuids::uuid, error_function_updater_cuda_smart_ptr> sample_map;
			sample_map sample_error_function_updater_cuda_map;
		};

		typedef boost::serialization::singleton<error_function_updater_cuda_factory> single_error_function_updater_cuda_factory;
	}
}
