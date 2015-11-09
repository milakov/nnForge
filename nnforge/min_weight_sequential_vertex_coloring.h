/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include <vector>
#include <boost/graph/graph_traits.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/limits.hpp>

namespace nnforge
{
	template <class VertexListGraph, class OrderPA, class ColorMap, class WeightPropertyMap>
	typename boost::property_traits<ColorMap>::value_type min_weight_sequential_vertex_coloring(const VertexListGraph& G, const OrderPA& order, ColorMap& color, const WeightPropertyMap& weight)
	{
		typedef boost::graph_traits<VertexListGraph> GraphTraits;
		typedef typename GraphTraits::vertex_descriptor Vertex;
		typedef typename boost::property_traits<ColorMap>::value_type size_type;
		typedef typename boost::property_traits<WeightPropertyMap>::value_type weight_type;
    
		size_type max_color = 0;
		const size_type V = boost::num_vertices(G);

		// We need to keep track of which colors are used by
		// adjacent vertices. We do this by marking the colors
		// that are used. The mark array contains the mark
		// for each color. The length of mark is the
		// number of vertices since the maximum possible number of colors
		// is the number of vertices.
		std::vector<size_type> mark(V, std::numeric_limits<size_type>::max());
		std::vector<weight_type> color_set_max_weight(V, ((!std::numeric_limits<weight_type>::is_signed) || std::numeric_limits<weight_type>::is_integer) ? std::numeric_limits<weight_type>::min() : -std::numeric_limits<weight_type>::max());
    
		//Initialize colors 
		typename GraphTraits::vertex_iterator v, vend;
		for (boost::tie(v, vend) = boost::vertices(G); v != vend; ++v)
			put(color, *v, V-1);
    
		//Determine the color for every vertex one by one
		for (size_type i = 0; i < V; i++)
		{
			Vertex current = boost::get(order,i);
			typename GraphTraits::adjacency_iterator v, vend;
      
			//Mark the colors of vertices adjacent to current.
			//i can be the value for marking since i increases successively
			for (boost::tie(v,vend) = boost::adjacent_vertices(current, G); v != vend; ++v)
				mark[boost::get(color,*v)] = i; 
      
			// Scan through all currently allocated color
			// Find the one with weight closest to the vertex weights, with preference to vertex weight fitting current color weight
			weight_type vertex_weight = boost::get(weight, current);
			bool color_found = false;
			size_type best_color;
			bool best_underfit;
			weight_type best_weight_diff;
			for(size_type current_color = 0; current_color < max_color; ++current_color)
			{
				if (mark[current_color] == i)
					continue;

				weight_type current_color_weight = color_set_max_weight[current_color];
				bool current_underfit;
				weight_type current_weight_diff;
				if (vertex_weight <= current_color_weight)
				{
					current_underfit = true;
					current_weight_diff = current_color_weight - vertex_weight;
				}
				else
				{
					current_underfit = false;
					current_weight_diff = vertex_weight - current_color_weight;
				}

				bool update_color = !color_found;
				if (!update_color)
				{
					if (best_underfit)
					{
						if (current_underfit)
							update_color = current_weight_diff < best_weight_diff;
						else
							update_color = false;
					}
					else
					{
						if (current_underfit)
							update_color = true;
						else
							update_color = current_weight_diff < best_weight_diff;
					}
				}

				if (update_color)
				{
					color_found = true;
					best_color = current_color;
					best_underfit = current_underfit;
					best_weight_diff = current_weight_diff;
				}
			}
      
			if (!color_found)  //All colors are used up. Add one more color
			{
				best_color = max_color;
				++max_color;
			}

			boost::put(color, current, best_color); // Save the color of the current vertex
			color_set_max_weight[best_color] = std::max(color_set_max_weight[best_color], vertex_weight); // Upate the max weight for the color
		}
    
		return max_color;
	}
}
