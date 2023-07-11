package training

import (
	messages "agentske/proto"
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
)

func ConvertToGlobalWeights(n *MLP) *messages.GlobalWeights {
	globalWeights := &messages.GlobalWeights{}

	// Convert biases
	for _, bias := range n.GetBiases() {
		biasData := &messages.Biases{
			Data: bias.RawMatrix().Data,
		}
		globalWeights.Biases = append(globalWeights.Biases, biasData)
	}

	// Convert weights
	for _, weight := range n.GetWeights() {
		weightData := &messages.Weights{
			Data: weight.RawMatrix().Data,
		}
		globalWeights.Weights = append(globalWeights.Weights, weightData)
	}

	return globalWeights
}

func (n *MLP) WriteWeightsToFile(filename string) error {
	// Create a struct to hold the weights
	weightsData := struct {
		Biases  [][]float64 `json:"biases"`
		Weights [][]float64 `json:"weights"`
	}{
		Biases:  make([][]float64, len(n.Biases)),
		Weights: make([][]float64, len(n.Weights)),
	}

	// Convert biases and weights to slices of slices of float64
	for i := 0; i < len(n.Biases); i++ {
		weightsData.Biases[i] = n.Biases[i].RawMatrix().Data
		weightsData.Weights[i] = n.Weights[i].RawMatrix().Data
	}

	// Serialize the weights data to JSON
	jsonData, err := json.MarshalIndent(weightsData, "", "  ")
	if err != nil {
		return err
	}

	// Write the JSON data to the file
	err = ioutil.WriteFile(filename, jsonData, 0644)
	if err != nil {
		return err
	}

	return nil
}

func (n *MLP) ReadWeightsFromFile(filename string) error {
	// Read the JSON data from the file
	jsonData, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}

	// Create a struct to hold the weights data
	weightsData := struct {
		Biases  [][]float64 `json:"biases"`
		Weights [][]float64 `json:"weights"`
	}{}

	// Unmarshal the JSON data into the weights data struct
	err = json.Unmarshal(jsonData, &weightsData)
	if err != nil {
		return err
	}

	// Convert the weights data into biases and weights matrices
	n.Biases = make([]*mat.Dense, len(weightsData.Biases))
	n.Weights = make([]*mat.Dense, len(weightsData.Weights))

	for i := 0; i < len(weightsData.Biases); i++ {
		biasesData := weightsData.Biases[i]
		weightsData := weightsData.Weights[i]

		biases := mat.NewDense(len(biasesData), 1, biasesData)
		weights := mat.NewDense(len(weightsData)/len(biasesData), len(biasesData), weightsData)

		n.Biases[i] = biases
		n.Weights[i] = weights
	}

	return nil
}

func (n *MLP) ConvertFromGlobalWeights(globalWeights *messages.GlobalWeights) {
	// Clear existing biases and weights
	n.Biases = []*mat.Dense{}
	n.Weights = []*mat.Dense{}

	// Convert biases
	for _, biasesData := range globalWeights.Biases {
		rows := len(biasesData.Data)
		cols := 1
		biases := mat.NewDense(rows, cols, biasesData.Data)
		n.Biases = append(n.Biases, biases)
	}

	// Convert weights
	for i, weightsData := range globalWeights.Weights {
		rows := n.sizes[i+1]
		cols := n.sizes[i]
		weights := mat.NewDense(cols, rows, weightsData.Data)
		n.Weights = append(n.Weights, weights)
	}

}
