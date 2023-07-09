package training

import (
	messages "agentske/proto"
	"fmt"
	"github.com/asynkron/protoactor-go/actor"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
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

func (n *MLP) backward(x, y mat.Matrix, context actor.Context, coordinationActor *actor.PID) {

	//mozda treba da se poveca vreme odziva
	fmt.Println(coordinationActor)
	aggregationActor, _ := context.RequestFuture(coordinationActor, &messages.GetAggregationActor{}, 1*time.Second).Result()
	fmt.Println(aggregationActor)
	globalWeights, _ := context.RequestFuture(aggregationActor.(*actor.PID), &messages.GetGlobalWeights{}, 1*time.Second).Result()
	fmt.Println(globalWeights)

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

// get prediction as max prob in row
func prediction(vs []float64) float64 {
	if vs[0] < 0.5 {
		return 0.0
	} else {
		return 1.0
	}
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
	n.Train(X, Y, context, coordinationActor)
	accuracy := n.Evaluate(Xv, Yv)
	fmt.Printf("accuracy = %0.1f%%\n", accuracy)
}
