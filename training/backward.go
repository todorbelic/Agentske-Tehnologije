package training

import (
	messages "agentske/proto"
	"github.com/asynkron/protoactor-go/actor"
	"gonum.org/v1/gonum/mat"
	"time"
)

func (n *MLP) Backward(x, y mat.Matrix, context actor.Context) {

	//mozda treba da se poveca vreme odziva
	aggregationActor, _ := context.RequestFuture(context.Parent(), &messages.GetAggregationActor{}, 5*time.Second).Result()
	globalWeightsResult, _ := context.RequestFuture(aggregationActor.(*actor.PID), &messages.GetGlobalWeights{}, 20*time.Second).Result()
	globalWeights := globalWeightsResult.(*messages.GlobalWeights)
	n.ConvertFromGlobalWeights(globalWeights)

	// get activations
	as, zs := n.Forward(x)

	// final z
	z := zs[len(zs)-1]
	out := as[len(as)-1]

	// error
	err := new(mat.Dense)

	err.Sub(out, y)

	// delta of last layer
	// delta = (out - y).sigmoidprime(last_z)
	sp := new(mat.Dense)
	sp.Apply(ApplySigmoidPrime, z)

	delta := new(mat.Dense)
	delta.MulElem(err, sp)

	// prop delta through layers

	nbs := make([]*mat.Dense, len(n.Weights))
	nws := make([]*mat.Dense, len(n.Weights))

	nbs[len(nbs)-1] = delta

	a := as[len(as)-2]

	nw := new(mat.Dense)
	nw.Mul(a.T(), delta)
	nws[len(nws)-1] = nw

	gradientsMsg := &messages.GradientUpdate{
		Weights: make([]*messages.WeightLayer, len(n.Weights)),
	}
	weightLayerLast := &messages.WeightLayer{
		Weights: nw.RawMatrix().Data,
		Biases:  SumCols(delta).RawMatrix().Data,
	}
	gradientsMsg.Weights[len(n.Weights)-1] = weightLayerLast

	// go back through layers
	for i := n.numLayers - 2; i > 0; i-- {
		z := zs[i-1] // -1?

		sp := new(mat.Dense)
		sp.Apply(ApplySigmoidPrime, z)

		wdelta := new(mat.Dense)
		w := n.Weights[i]

		wdelta.Mul(delta, w.T())

		nextdelta := new(mat.Dense)
		nextdelta.MulElem(wdelta, sp)
		delta = nextdelta

		nbs[i-1] = delta

		a := as[i-1]
		nw := new(mat.Dense)
		nw.Mul(a.T(), delta)
		nws[i-1] = nw

		//r, c := nbs[i-1].Dims()
		//fmt.Println("Nws: ", r, c)

		weightLayer := &messages.WeightLayer{
			Weights: nws[i-1].RawMatrix().Data,
			Biases:  SumCols(nbs[i-1]).RawMatrix().Data,
		}

		gradientsMsg.Weights[i-1] = weightLayer
	}

	N, _ := x.Dims()
	gradientsMsg.BatchSize = int32(N)

	context.Send(aggregationActor.(*actor.PID), gradientsMsg)
}
