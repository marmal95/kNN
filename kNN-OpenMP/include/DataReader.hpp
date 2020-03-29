#pragma once

#include "DataRow.hpp"
#include "FlatDataView.hpp"
#include <fstream>
#include <string_view>

struct LabelIndex
{
	LabelIndex(const std::size_t labelIndex) : value{ labelIndex } {}
	const std::size_t value;
	const static inline auto LAST = std::numeric_limits<std::size_t>::max();
};

class DataReader
{
public:
	DataReader(std::string_view, const LabelIndex, const char delimiter);
	~DataReader();

	std::vector<DataRow> readData();
	FlatDataView readDataFlat();

private:
	DataRow createDataRow(const std::string&) const;
	void skipDelimiter(std::istringstream&) const;
	const std::size_t getLabelIndex(const std::size_t) const;

	std::ifstream inputFile;
	const LabelIndex labelIndex;
	const char delimiter;
};