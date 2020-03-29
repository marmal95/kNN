#include "Accuracy.hpp"

std::ostream& operator<<(std::ostream& os, const Accuracy& accuracy)
{
	return os << accuracy.correct << " / " << accuracy.all;
}
