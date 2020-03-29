#include "DataReader.hpp"
#include <sstream>
#include <iostream>

DataReader::DataReader(std::string_view filename, const LabelIndex labelIndex, const char delimiter)
	: inputFile{ filename.data(), std::fstream::in }, labelIndex{ labelIndex }, delimiter{ delimiter }
{}

DataReader::~DataReader()
{
	if (inputFile.is_open())
	{
		inputFile.close();
	}
}

std::vector<DataRow> DataReader::readData()
{
	inputFile.clear();
	inputFile.seekg(0);

	std::vector<DataRow> rows{};
	std::string line{};

	while (std::getline(inputFile, line))
	{
		rows.emplace_back(createDataRow(line));
	}

	return rows;
}

FlatDataView DataReader::readDataFlat()
{
	return { readData() };
}

DataRow DataReader::createDataRow(const std::string& rowData) const
{
	std::istringstream iss{ rowData };

	float feature{};
	char delimiter{};
	DataRow row{};

	while (iss >> feature)
	{
		row.features.push_back(feature);
		skipDelimiter(iss);
	}

	const std::size_t labelIndex = getLabelIndex(row.features.size());
	row.label = row.features[labelIndex];
	row.features.erase(row.features.begin() + labelIndex);
	return row;
}

void DataReader::skipDelimiter(std::istringstream& iss) const
{
	if (this->delimiter != ' ')
	{
		char delimiter;
		iss >> delimiter;
	}
}

const std::size_t DataReader::getLabelIndex(const std::size_t rowSize) const
{
	return labelIndex.value == LabelIndex::LAST ? rowSize - 1 : labelIndex.value;
}
