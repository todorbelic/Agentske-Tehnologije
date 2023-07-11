package training

import (
	messages "agentske/proto"
	"gonum.org/v1/gonum/mat"
)

func UpdateGlobalWeights(n *MLP, msg *messages.GradientUpdate) {
	for i, weightLayer := range msg.Weights {
		wrows, wcols := n.Weights[i].Dims()
		brows, bcols := n.Biases[i].Dims()
		nw := mat.NewDense(wrows, wcols, weightLayer.Weights)
		nb := mat.NewDense(bcols, brows, weightLayer.Biases).T()
		alpha := 0.3 / float64(msg.BatchSize)

		scalednw := new(mat.Dense)
		scalednw.Scale(alpha, nw)

		scalednb := new(mat.Dense)
		scalednb.Scale(alpha, nb)

		wprime := new(mat.Dense)
		wprime.Sub(n.Weights[i], scalednw)

		bprime := new(mat.Dense)
		bprime.Sub(n.Biases[i], scalednb)

		n.Weights[i] = wprime
		n.Biases[i] = bprime
	}
}
