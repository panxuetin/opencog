/*
 * opencog/learning/statistics/Entropy.cc
 *
 * Copyright (C) 2012 by OpenCog Foundation
 * All Rights Reserved
 *
 * Written by Shujing Ke <rainkekekeke@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "Entropy.h"
#include <map>
#include <iterator>
#include <math.h>

using namespace opencog::statistics;
using namespace std;

template<typename Metadata>
void Entropy::calculateEntropies(DataProvider<Metadata>* data)
{
    map<string,StatisticData>::iterator it;

    for (int n = 1; n <= data->n_gram; ++n )
    {
        for( it = data->mDataMaps[n].begin(); it < data->mDataMaps[n].end(); ++it)
        {
            StatisticData& pieceData = (StatisticData)(it->second);
            pieceData.entropy = (-1.0f) * pieceData.probability * log2(pieceData.probability);

        }

    }
}

template<typename Metadata>
void Entropy::calculateProbabilityAndEntropies(DataProvider<Metadata>* data)
{
    map<string,StatisticData>::iterator it;

    for (int n = 1; n <= data->n_gram; ++n )
    {
        for( it = data->mDataMaps[n].begin(); it < data->mDataMaps[n].end(); ++it)
        {
            StatisticData& pieceData = (StatisticData)(it->second);
            pieceData.probability = ((float)(pieceData.count))/((float)(data->mRawDataNumbers[n]));
            pieceData.entropy = (-1.0f) * pieceData.probability * log2(pieceData.probability);
        }
    }
}

