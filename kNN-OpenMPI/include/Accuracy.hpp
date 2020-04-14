#pragma once

#include <cstdlib>
#include <iostream>

struct Accuracy
{
    std::size_t correct;
    std::size_t all;
};

std::ostream& operator<<(std::ostream& os, const Accuracy&);
