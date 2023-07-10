package training

import (
	messages "agentske/proto"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"time"

	"github.com/asynkron/protoactor-go/actor"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Config struct {
	Epochs    int
	BatchSize int
	Eta       float64
}

type MLP struct {
	numLayers int
	sizes     []int
	biases    []*mat.Dense
	weights   []*mat.Dense
	config    Config
}

func (n *MLP) GetWeights() []*mat.Dense {
	return n.weights
}

func (n *MLP) GetBiases() []*mat.Dense {
	return n.biases
}

func New(c Config, sizes ...int) *MLP {

	// generate some random weights and biases
	bs := []*mat.Dense{}
	ws := []*mat.Dense{}

	// len of slices we will make
	// don't need any biases for input layer
	// don't need any weights for output layer
	l := len(sizes) - 1

	for j := 0; j < l; j++ {
		y := sizes[1:][j] // y starts from layer after input layer to output layer
		x := sizes[:l][j] // x starts from input layer to layer before output layer

		// make a random init biases matrix of y*1
		b := make([]float64, y)
		for i := range b {
			b[i] = rand.NormFloat64()
		}
		bs = append(bs, mat.NewDense(y, 1, b))

		// make a random init weights matrix of y*x
		w := make([]float64, y*x)
		for i := range w {
			w[i] = rand.NormFloat64()
		}
		ws = append(ws, mat.NewDense(x, y, w)) // P:changed the order of row and column

	}

	return &MLP{
		numLayers: len(sizes),
		sizes:     sizes,
		biases:    bs,
		weights:   ws,
		config:    c,
	}
}

func (n *MLP) WriteWeightsToFile(filename string) error {
	// Create a struct to hold the weights
	weightsData := struct {
		Biases  [][]float64 `json:"biases"`
		Weights [][]float64 `json:"weights"`
	}{
		Biases:  make([][]float64, len(n.biases)),
		Weights: make([][]float64, len(n.weights)),
	}

	// Convert biases and weights to slices of slices of float64
	for i := 0; i < len(n.biases); i++ {
		weightsData.Biases[i] = n.biases[i].RawMatrix().Data
		weightsData.Weights[i] = n.weights[i].RawMatrix().Data
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
	n.biases = make([]*mat.Dense, len(weightsData.Biases))
	n.weights = make([]*mat.Dense, len(weightsData.Weights))

	for i := 0; i < len(weightsData.Biases); i++ {
		biasesData := weightsData.Biases[i]
		weightsData := weightsData.Weights[i]

		biases := mat.NewDense(len(biasesData), 1, biasesData)
		weights := mat.NewDense(len(weightsData)/len(biasesData), len(biasesData), weightsData)

		n.biases[i] = biases
		n.weights[i] = weights
	}

	return nil
}

func sumCols(m *mat.Dense) *mat.Dense {

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
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func applySigmoid(_, _ int, v float64) float64 {
	return sigmoid(v)
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func applySigmoidPrime(_, _ int, v float64) float64 {
	return sigmoidPrime(v)
}

func (n *MLP) forward(x mat.Matrix) (as, zs []mat.Matrix) {

	as = append(as, x)
	_x := x

	for i := 0; i < len(n.weights); i++ {

		w := n.weights[i]
		b := n.biases[i]

		// z = w.x + b

		m := new(mat.Dense)

		m.Mul(_x, w)

		z := new(mat.Dense)
		addB := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
		z.Apply(addB, m)

		zs = append(zs, z)

		// a = sigmoid(z)
		a := new(mat.Dense)
		a.Apply(applySigmoid, z)
		as = append(as, a)

		_x = a
	}

	return

}

func (n *MLP) ConvertFromGlobalWeights(globalWeights *messages.GlobalWeights) {
	// Clear existing biases and weights
	n.biases = []*mat.Dense{}
	n.weights = []*mat.Dense{}

	// Convert biases
	for _, biasesData := range globalWeights.Biases {
		rows := len(biasesData.Data)
		cols := 1
		biases := mat.NewDense(rows, cols, biasesData.Data)
		n.biases = append(n.biases, biases)
	}

	// Convert weights
	for i, weightsData := range globalWeights.Weights {
		rows := n.sizes[i+1]
		cols := n.sizes[i]
		weights := mat.NewDense(cols, rows, weightsData.Data)
		n.weights = append(n.weights, weights)
	}

}

func (n *MLP) backward(x, y mat.Matrix, context actor.Context, coordinationActor *actor.PID) {

	//mozda treba da se poveca vreme odziva
	fmt.Println(coordinationActor)
	aggregationActor, _ := context.RequestFuture(coordinationActor, &messages.GetAggregationActor{}, 5*time.Second).Result()
	fmt.Println("Aggregation: ", aggregationActor)
	globalWeightsResult, _ := context.RequestFuture(aggregationActor.(*actor.PID), &messages.GetGlobalWeights{}, 10*time.Second).Result()
	globalWeights := globalWeightsResult.(*messages.GlobalWeights)
	// fmt.Println("Global:", globalWeights)
	n.ConvertFromGlobalWeights(globalWeights)

	// get activations
	as, zs := n.forward(x)

	// final z
	z := zs[len(zs)-1]
	out := as[len(as)-1]

	// error
	err := new(mat.Dense)

	err.Sub(out, y)

	// delta of last layer
	// delta = (out - y).sigmoidprime(last_z)
	sp := new(mat.Dense)
	sp.Apply(applySigmoidPrime, z)

	delta := new(mat.Dense)
	delta.MulElem(err, sp)

	// prop delta through layers

	nbs := make([]*mat.Dense, len(n.weights))
	nws := make([]*mat.Dense, len(n.weights))

	nbs[len(nbs)-1] = delta

	a := as[len(as)-2]

	nw := new(mat.Dense)
	nw.Mul(a.T(), delta)
	nws[len(nws)-1] = nw

	gradientsMsg := &messages.GradientUpdate{
		Weights: make([]*messages.WeightLayer, len(n.weights)),
	}

	// go back through layers
	for i := n.numLayers - 2; i > 0; i-- {
		z := zs[i-1] // -1?

		sp := new(mat.Dense)
		sp.Apply(applySigmoidPrime, z)

		wdelta := new(mat.Dense)
		w := n.weights[i]

		wdelta.Mul(delta, w.T())

		nextdelta := new(mat.Dense)
		nextdelta.MulElem(wdelta, sp)
		delta = nextdelta

		nbs[i-1] = delta

		a := as[i-1]
		nw := new(mat.Dense)
		nw.Mul(a.T(), delta)
		nws[i-1] = nw

		r, c := nws[i-1].Dims()
		fmt.Println("Nws: ", r, c)

		weightLayer := &messages.WeightLayer{
			Weights: nws[i-1].RawMatrix().Data,
			Biases:  nbs[i-1].RawMatrix().Data,
		}

		gradientsMsg.Weights[i-1] = weightLayer
	}

	N, _ := x.Dims()

	weights := make([]*mat.Dense, len(n.weights))
	biases := make([]*mat.Dense, len(n.biases))

	for i := 0; i < len(n.weights); i++ {
		w := n.weights[i]
		nw := nws[i]

		b := n.biases[i]
		nb := sumCols(nbs[i]).T()

		// w' = w - (eta/N) * nw

		alpha := n.config.Eta / float64(N)
		scalednw := new(mat.Dense)
		scalednw.Scale(alpha, nw)

		scalednb := new(mat.Dense)
		scalednb.Scale(alpha, nb)

		wprime := new(mat.Dense)
		wprime.Sub(w, scalednw)

		bprime := new(mat.Dense)
		bprime.Sub(b, scalednb)

		weights[i] = wprime
		biases[i] = bprime

	}

	n.weights = weights
	n.biases = biases

	context.Send(aggregationActor.(*actor.PID), gradientsMsg)
}

func (n *MLP) Predict(x mat.Matrix) mat.Matrix {

	as, _ := n.forward(x)

	return as[len(as)-1]
}

func (n *MLP) Train(x, y *mat.Dense, context actor.Context, coordinationActor *actor.PID) {

	r, cx := x.Dims()
	_, cy := y.Dims()

	b := n.config.BatchSize

	for e := 1; e < n.config.Epochs+1; e++ {

		for i := 0; i < r; i += b {
			k := i + b
			if k > r {
				k = r
			}
			_x := x.Slice(i, k, 0, cx)
			_y := y.Slice(i, k, 0, cy)

			n.backward(_x, _y, context, coordinationActor)
		}
	}
}

func (n *MLP) Evaluate(x, y mat.Matrix) float64 {

	p := n.Predict(x)
	N, _ := p.Dims()

	var correct int
	for n := 0; n < N; n++ {

		ry := mat.Row(nil, n, y)
		truth := ry[0]

		rp := mat.Row(nil, n, p)
		predicted := prediction(rp)

		if predicted == truth {
			correct++
		}
	}

	accuracy := float64(correct) / float64(N)

	return accuracy * 100
}

func (n *MLP) EvaluateRecall(x, y mat.Matrix) float64 {
	p := n.Predict(x)
	N, _ := p.Dims()

	var truePositives int
	var actualPositives int

	for i := 0; i < N; i++ {
		ry := mat.Row(nil, i, y)
		truth := ry[0]

		rp := mat.Row(nil, i, p)
		predicted := prediction(rp)

		if predicted == 1.0 && truth == 1.0 {
			truePositives++
		}

		if truth == 1.0 {
			actualPositives++
		}
	}

	if actualPositives == 0 {
		return 0.0
	}

	recall := float64(truePositives) / float64(actualPositives)
	return recall * 100
}

// get prediction as max prob in row
func prediction(vs []float64) float64 {
	if vs[0] < 0.5 {
		return 0.0
	} else {
		return 1.0
	}
}

func EvaluateF1Score(n *MLP, x, y mat.Matrix) float64 {
	precision := n.Evaluate(x, y)
	recall := n.EvaluateRecall(x, y)

	if precision+recall == 0.0 {
		return 0.0
	}

	f1Score := 2 * (precision * recall) / (precision + recall)
	return f1Score
}

func StartTraining(X, Y, Xv, Yv *mat.Dense, context actor.Context, coordinationActor *actor.PID) {
	con := Config{
		Epochs:    25,
		Eta:       0.3,
		BatchSize: 32,
	}
	_, cols := X.Dims()
	arch := []int{cols, 15, 8, 1}
	n := New(con, arch...)
	n.WriteWeightsToFile("weights.json")
	fmt.Println("Wrote")
	n.Train(X, Y, context, coordinationActor)
	accuracy := n.Evaluate(Xv, Yv)
	recall := n.EvaluateRecall(Xv, Yv)
	fmt.Printf("accuracy = %0.1f%%\n", accuracy)
	fmt.Printf("recall = %0.1f%%", recall)
}
