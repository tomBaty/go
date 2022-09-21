package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

func readData(filename string, features int) {

	train, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	defer train.Close()

	csvReader := csv.NewReader(train)
	data, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	all_train_instances := make([][]float64, 0)

	for i, line := range data {
		// skip header line of csv files
		if i == 0 {
			continue
		}
		line_floats := make([]float64, 0)
		individual_ams := strings.Fields(line[0])
		for _, s := range individual_ams {
			if n, err := strconv.ParseFloat(s, 64); err == nil {
				line_floats = append(line_floats, n)
			}
		}

		all_train_instances = append(all_train_instances, line_floats)

	}
	fmt.Print(all_train_instances[0])
}

func calculateDistance() {

}

func predictInstance() {

}

func main() {
	// var cl_args_len = len(os.Args)
	// if cl_args_len != 4 {
	// 	panic("Required arguments: train file name, test file name, num. features")
	// }
	var train_name string = os.Args[1]
	// var test_name string = os.Args[2]
	// var features string = os.Args[3] // 14

	readData(train_name, 14)

}
