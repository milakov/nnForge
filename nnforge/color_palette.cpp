#include "color_palette.h"

#include <boost/format.hpp>
#include <algorithm>
#include <limits>

namespace nnforge
{
	unsigned int get_distance_squared(const rgba_color& x, const rgba_color& y)
	{
		unsigned int sum = 0;
		for(int i = 0; i < 4; ++i)
			sum += (x.c.rgba[i] - y.c.rgba[i]) * (x.c.rgba[i] - y.c.rgba[i]);
		return sum;
	}

	color_palette::color_palette()
	{
		colors.push_back(0x8EDFBF);
		colors.push_back(0xD4E6E8);
		colors.push_back(0xF4BE72);
		colors.push_back(0x5F9F74);
		colors.push_back(0xEC99B7);
		colors.push_back(0xD2F08A);
		colors.push_back(0x8096AE);
		colors.push_back(0xA39770);
		colors.push_back(0xCAFFD1);
		colors.push_back(0xEB9986);
		colors.push_back(0xEDC9FB);
		colors.push_back(0x6FC9C9);
		colors.push_back(0xFFD3D4);
		colors.push_back(0xF3ECC3);
		colors.push_back(0xA89595);
		colors.push_back(0x86D9A0);
		colors.push_back(0x979B53);
		colors.push_back(0xFAE693);
		colors.push_back(0xAFFBF8);
		colors.push_back(0xA2C573);
		colors.push_back(0xA4DBF1);
		colors.push_back(0xDCF9AE);
		colors.push_back(0xA4BADC);
		colors.push_back(0xEEB6D8);
		colors.push_back(0x64A48F);
		colors.push_back(0xFDC390);
		colors.push_back(0xC98564);
		colors.push_back(0xC4F2DB);
		colors.push_back(0x879E65);
		colors.push_back(0xF9B79B);
		colors.push_back(0xCB8B9E);
		colors.push_back(0xB6ED93);
		colors.push_back(0xBAC16E);
		colors.push_back(0xD0BDEA);
		colors.push_back(0xD9867A);

		for(unsigned int colors_chosen_count = 1; colors_chosen_count < static_cast<unsigned int>(colors.size() - 1); ++colors_chosen_count)
		{
			unsigned int best_distant_color_pos = colors_chosen_count;
			unsigned int best_min_distance_squared = 0;
			for(unsigned int candidate_distant_color_pos = colors_chosen_count; candidate_distant_color_pos < static_cast<unsigned int>(colors.size()); ++candidate_distant_color_pos)
			{
				unsigned int min_distance_squared = std::numeric_limits<unsigned int>::max();
				for(unsigned int chosen_pos = 0; chosen_pos < colors_chosen_count; ++chosen_pos)
				{
					unsigned int new_distance = get_distance_squared(colors[candidate_distant_color_pos], colors[chosen_pos]);
					min_distance_squared = std::min(min_distance_squared, new_distance);
				}
				if (min_distance_squared > best_min_distance_squared)
				{
					best_distant_color_pos = candidate_distant_color_pos;
					best_min_distance_squared = min_distance_squared;
				}
			}

			std::swap(colors[colors_chosen_count], colors[best_distant_color_pos]);
		}
	}

	std::string color_palette::get_color_name(unsigned int logical_color_id) const
	{
		unsigned int color_id_wrapped = (logical_color_id % colors.size());
		return (boost::format("#%|1$06x|") % colors[color_id_wrapped].c.val).str();
	}

	color_palette& color_palette::get_singleton()
	{
		static color_palette instance;
		return instance;
	}
}
