package proto_conversion

import (
	"agentske/preprocessing"
	messages "agentske/proto"
	"gonum.org/v1/gonum/mat"
)

func ConvertToProtoData(data preprocessing.Data) (*messages.Data, error) {
	protoData := &messages.Data{}

	// Assign Labels
	protoData.Labels = ConvertToInt32Slice(data.Labels)

	// Convert Histograms
	for _, hist := range data.Histograms {
		protoHist := &messages.Histogram{}
		protoHist.Values = hist

		protoData.Histograms = append(protoData.Histograms, protoHist)
	}

	return protoData, nil
}

func ConvertToInt32Slice(labels []float64) []float64 {
	result := make([]float64, len(labels))
	for i, val := range labels {
		result[i] = float64(val)
	}
	return result
}

func GetDataSetsFromProto(data *messages.Data) (*mat.Dense, *mat.Dense, error) {
	rows, cols := len(data.Histograms), len(data.Histograms[0].Values)
	trainingData := make([]float64, rows*cols)
	for i, hist := range data.Histograms {
		copy(trainingData[i*cols:(i+1)*cols], hist.Values)
	}
	X := mat.NewDense(rows, cols, trainingData)
	Y := mat.NewDense(rows, 1, data.Labels)
	return X, Y, nil
}
