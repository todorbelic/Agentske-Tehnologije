package training

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
)

func SumCols(m *mat.Dense) *mat.Dense {

	_, c := m.Dims()

	var output *mat.Dense

	data := make([]float64, c)
	for i := 0; i < c; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	output = mat.NewDense(1, c, data)

	return output
}

// sigmoid is an elementwise func
// this applied over every element
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func ApplySigmoid(_, _ int, v float64) float64 {
	return Sigmoid(v)
}

func Sigmoidprime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func ApplySigmoidPrime(_, _ int, v float64) float64 {
	return Sigmoidprime(v)
}
