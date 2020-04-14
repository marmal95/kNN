#pragma once

#include "FlatDataView.hpp"

namespace Cuda
{
    void knn(const FlatDataView&, FlatDataView&);
    void minMax(FlatDataView&);
}