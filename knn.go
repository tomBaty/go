package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

var train_instances []KInstance
var ranges []float64

// An instance used for training or predicting with KNearestNeighbours.
//
//	values: float64 values for each feature.
//	label: what class this instance is.
type KInstance struct {
	values []float64
	label  float64
}

// Calculate minimum and maximum value in an array/slice.
//
//	array: slice/array to find values in.
func minAndMaxOfArray(array []float64) (min float64, max float64) {

	min = array[0]
	max = array[0]
	for _, value := range array {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}
	return min, max
}

// Read in data from a csv file and convert it to a list of KInstances.
//
//	filename: filepath of file.
//	features: number of columns in the csv file
//	training_set: if True, calculate ranges.
func readData(filename string, features int, trainingSet bool) []KInstance {

	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	// close file when program finishes
	defer f.Close()

	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		panic(err)
	}

	instances := make([]KInstance, 0)

	for i, line := range data {
		// skip header line of csv files
		if i == 0 {
			continue
		}
		line_floats := make([]float64, 0)
		individual_ams := strings.Fields(line[0])
		// reader reads in strings, so they have to be converted to floats with ParseFloat()
		for _, s := range individual_ams {
			if n, err := strconv.ParseFloat(s, 64); err == nil {
				line_floats = append(line_floats, n)
			}
		}

		label := line_floats[len(line_floats)-1]
		values := line_floats[0 : len(line_floats)-1]
		// store as a KInstance struct
		instances = append(instances, KInstance{values, label})
	}

	for j := 0; j < features-1; j++ {
		column_values := make([]float64, 0, len(instances))
		// ranges are used for distance calculations, necessary to calculate for train data
		for i := range instances {
			value := instances[i].values[j]
			column_values = append(column_values, value)
		}

		min, max := minAndMaxOfArray(column_values)
		ranges = append(ranges, (max - min))

	}

	return instances
}

// Calculate mean distance across features between train instance a and test instance b
func distance(a KInstance, b KInstance) float64 {
	summed_distance := 0.0
	for i := range a.values {
		summed_distance += math.Pow(a.values[i]-b.values[i], 2) / math.Pow(ranges[i], 2)
	}
	return math.Sqrt(summed_distance)
}

// Bundle together values with label so they can be sorted and retain their labels.
type distanceWithTag struct {
	distanceValue float64
	label         float64
}

// Predict a single test instance.
//
//	k: Number of neighbours to compare when predicting.
//	test_instance: Instance to predoct a label for.
func predictInstance(k int, test_instance KInstance) float64 {

	distances := make([]distanceWithTag, 0)

	for _, instance := range train_instances {
		between := distance(instance, test_instance)
		distances = append(distances, distanceWithTag{between, instance.label})
	}
	// sort distances so we can find the closest neighbours
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distanceValue < distances[j].distanceValue
	})
	// we only need the k closest neighbours
	distances = distances[:k]

	neighbours := make(map[float64]int)
	for _, instance := range distances {
		// check if already in map
		if val, inMap := neighbours[instance.label]; inMap {
			// already in map, increment
			neighbours[instance.label] = val + 1
		} else {
			// not in map, set to 1
			neighbours[instance.label] = 1
		}
	}
	keys := make([]float64, 0, len(neighbours))
	// get list of keys
	for key := range neighbours {
		keys = append(keys, key)
	}
	// sort to get label with highest amount of neighbours
	sort.SliceStable(keys, func(i, j int) bool {
		return neighbours[keys[i]] > neighbours[keys[j]]
	})
	return keys[0]

}

// Predict labels for a test set based on a training set of instances. All features must be floats.
// Parameters are taken through command line:
//
//	train_name: filepath of the training data
//	test_name: filepath of the data to predict
//	features: amount of features of the data (columns)
//	k: amount of neighbours used in predictions (smaller tends to be better)
func main() {

	var cl_args_len = len(os.Args)
	if cl_args_len != 5 {
		panic("Required arguments: train file name, test file name, num. features, k")
	}
	var train_name string = os.Args[1]
	var test_name string = os.Args[2]
	// features for wine data: 14
	features, err := strconv.Atoi(os.Args[3])
	if err != nil {
		panic(err)
	}
	k, err := strconv.Atoi(os.Args[4])
	if err != nil {
		panic(err)
	}

	train_instances = readData(train_name, features, true)
	test_instances := readData(test_name, features, false)

	correct := 0.0
	for index, instance := range test_instances {
		if predictInstance(k, instance) == test_instances[index].label {
			correct += 1.0
		}
	}
	acc := correct / float64(len(test_instances))
	fmt.Printf("Classified %.0f/%d test instances correctly for an accuracy of %.6f", correct, len(test_instances), acc)
}
